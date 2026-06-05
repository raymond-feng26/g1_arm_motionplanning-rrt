"""
grasp_scene.py
==============
Shared perception -> scene -> IK-validation front-end for the grasp stack.

The two runners (grab_run.py, pull_left_run.py) share all of their
scene-building and IK-validation here, in ONE place:

  * snapshot_scene()     - wait for the box (id 0) + table (id 1) tags, build
                           T_base_box / T_base_table / grasp targets / the two
                           obstacle worlds, and hand them back as a `Scene`.
  * make_box_obb()       - tight oriented box (OBB) obstacle for the box.
  * make_table_slab()    - oriented table-slab obstacle, height tied to the box.
  * solve_and_validate() - 14-DOF IK + _validate (+ named collision report).
  * report_ee()          - print per-hand EE vs target error (print-only).
  * render_planned()     - write plotly HTML for the RRT legs AFTER a run.

Why this file exists
--------------------
The grab and pull runners need the same scene-building and IK-validation. This
module is the single source for it, so the runners read top-to-bottom with no
detours: each one is just `snapshot_scene -> validate -> build_* -> run`.

No hardware, no DDS: this module only talks to the perception socket (via a
caller-owned PerceptionClient) and the kinematic model. The caller owns the
G1Robot / G1DualArm and the client's lifecycle.
"""
import time
from dataclasses import dataclass

import numpy as np

from camera_calib import optical_to_base
from grasp_target import (
    box_in_base, table_in_base, grasp_targets,
    BOX_X_LEN, BOX_Y_LEN, BOX_Z_LEN,
    TABLE_THICK, TABLE_WIDTH, TABLE_DEPTH,
    TAG_TO_FAR_EDGE, TAG_TO_NEAR_EDGE,
)
# _validate / _default_obstacles are the planner's own goal checks and static
# world. Reuse them; do not reinvent. (g1_plan_run also owns plan() etc.)
from g1_plan_run import _validate, _default_obstacles, save_path_html


# --- defaults ---------------------------------------------------------------
PERCEPTION_HOST = "127.0.0.1"
PERCEPTION_PORT = 5000
BOX_TAG_ID      = 0          # tag on the box top face
TABLE_TAG_ID    = 1          # tag on the table (far-side middle)

# IK orientation defaults (torso frame). Identity = palm-forward along +x.
# The runners override these per-arm (R_left = Rz(box yaw), R_right = identity);
# the offline tests use the identity defaults below.
ORIENTATION_WEIGHT = 0.5
TARGET_LEFT_ORI    = np.eye(3)
TARGET_RIGHT_ORI   = np.eye(3)

# Obstacle-name labels, parallel to the obstacle lists (for collision_report
# readouts: "left_forearm vs box" instead of "obstacle 1"). _default_obstacles()
# is the single torso box, so these line up 1:1 with the worlds built below.
NAMES_FULL   = ["torso", "box", "table"]
NAMES_NO_BOX = ["torso", "table"]


# === scene geometry =========================================================

def make_box_obb(T_base_box):
    """Tight oriented box (OBB) for the box: [cx,cy,cz, roll,pitch,yaw, sx,sy,sz].

    box_in_base yaw-flattens the pose (pitch/roll = 0), so the box is a pure-yaw
    OBB: center = pose translation, yaw from the pose, sizes = the true box dims.
    No AABB inflation — the collision checker honors the yaw, so a yawed box is
    represented tightly instead of by its (much larger) enclosing AABB.

    Also returns the world-AABB corners (lo, hi) of that oriented box, for any
    caller that wants a plain bounding extent or unpacks three values.
    """
    R = T_base_box[:3, :3]
    c = T_base_box[:3, 3]
    yaw = np.arctan2(R[1, 0], R[0, 0])
    obs = [c[0], c[1], c[2], 0.0, 0.0, yaw, BOX_X_LEN, BOX_Y_LEN, BOX_Z_LEN]

    corners = np.array([[sx * BOX_X_LEN / 2, sy * BOX_Y_LEN / 2, sz * BOX_Z_LEN / 2]
                        for sx in (-1, +1)
                        for sy in (-1, +1)
                        for sz in (-1, +1)])
    corners_base = (R @ corners.T).T + c
    return obs, corners_base.min(axis=0), corners_base.max(axis=0)


def make_table_slab(T_base_table, table_top_z):
    """Table obstacle (oriented box).

    Takes ONLY depth + orientation (and x/y placement) from the table tag; the
    table-top HEIGHT comes from table_top_z = the box bottom (the box rests on
    the table). Tying the surface to the box bottom keeps the box sitting exactly
    on the table and sidesteps the box-tag vs table-tag z-calibration mismatch
    (the table tag's own z is intentionally ignored here).

    Depth (TABLE_DEPTH) spans near edge -> far edge along the table's +x; width
    is treated as very wide. The table's yaw goes in the rpy slots, so the
    OBB-aware checker represents a tilted table exactly.
    """
    # x/y of the slab center: depth-midpoint in the table frame, projected to
    # base. The tag's z is ignored — table height is set from the box bottom.
    cx_t = (TAG_TO_FAR_EDGE - TAG_TO_NEAR_EDGE) / 2.0
    center_xy = T_base_table @ np.array([cx_t, 0.0, 0.0, 1.0])
    cz = table_top_z - TABLE_THICK / 2.0          # top surface at the box bottom
    yaw = np.arctan2(T_base_table[1, 0], T_base_table[0, 0])
    return [center_xy[0], center_xy[1], cz,
            0.0, 0.0, yaw,
            TABLE_DEPTH, TABLE_WIDTH, TABLE_THICK]


# === scene snapshot =========================================================

@dataclass
class Scene:
    """A frozen snapshot of the world, everything the runners need to plan.

    T_base_box   : 4x4 box pose in base (yaw-only; pitch/roll zeroed).
    T_base_table : 4x4 table pose in base (yaw-only).
    targets      : dict from grasp_targets() — left/right pre + grasp + center.
    R_yaw        : 3x3 box rotation; used as the LEFT-hand approach orientation.
    box_obs      : the box OBB obstacle row.
    table_obs    : the table-slab obstacle row.
    obs_full     : world WITH the box   ([torso] + box + table) — approach leg.
    obs_no_box   : world WITHOUT the box ([torso] + table)       — grasp/lift.
    """
    T_base_box:   np.ndarray
    T_base_table: np.ndarray
    targets:      dict
    R_yaw:        np.ndarray
    box_obs:      list
    table_obs:    list
    obs_full:     list
    obs_no_box:   list

    @property
    def box_yaw_deg(self):
        return float(np.degrees(np.arctan2(self.T_base_box[1, 0], self.T_base_box[0, 0])))

    @property
    def table_yaw_deg(self):
        return float(np.degrees(np.arctan2(self.T_base_table[1, 0], self.T_base_table[0, 0])))

    def names_for(self, step):
        """Map a choreography step to its obstacle-name list by world size, so a
        preflight collision prints 'left_forearm vs torso' rather than 'obstacle
        0'. Passed straight to preflight_validate(..., obstacle_names=...)."""
        n = len(step.get("obstacles", []))
        return NAMES_NO_BOX if n == len(self.obs_no_box) else NAMES_FULL

    def print_summary(self):
        print("\n=== Scene ===")
        print(f"  box center: {self.T_base_box[:3, 3]}  yaw={self.box_yaw_deg:+.1f} deg")
        print(f"  table     : c={np.asarray(self.table_obs)[:3]}  "
              f"size={np.asarray(self.table_obs)[6:9]}  yaw={self.table_yaw_deg:+.1f} deg")


def snapshot_scene(client, box_id=BOX_TAG_ID, table_id=TABLE_TAG_ID,
                   timeout_s=10.0, poll_s=0.1):
    """Wait for both tags on an already-started client, build and return a `Scene`.

    `client` is a caller-owned, already-`start()`ed PerceptionClient. This polls
    up to `timeout_s` for BOTH the box tag and the table tag, then assembles the
    full scene (poses, grasp targets, the two obstacle worlds). Returns None if
    either tag never showed (the caller should abort).

    Lifecycle: this function does NOT start or stop the client — the caller owns
    it and keeps it alive (its read thread is a daemon, so it costs nothing to
    leave running and is torn down when the process exits). This is the single
    source for the tag-wait + scene-build that used to be copy-pasted in every
    grasp entry point.
    """
    print(f"[scene] waiting for box (id {box_id}) + table (id {table_id}) tags "
          f"(up to {timeout_s:.0f}s)...")
    T_opt_box = T_opt_table = None
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if T_opt_box   is None: T_opt_box,   _ = client.latest_pose(box_id)
        if T_opt_table is None: T_opt_table, _ = client.latest_pose(table_id)
        if T_opt_box is not None and T_opt_table is not None:
            break
        time.sleep(poll_s)

    if T_opt_box is None:
        print(f"[scene] no box tag (id {box_id}) — abort.")
        return None
    if T_opt_table is None:
        print(f"[scene] no table tag (id {table_id}) — abort.")
        return None

    T_base_box   = box_in_base(optical_to_base(T_opt_box))
    T_base_table = table_in_base(optical_to_base(T_opt_table))
    targets      = grasp_targets(T_base_box)
    R_yaw        = T_base_box[:3, :3]      # box yaw -> left-hand approach orientation

    box_obs, box_lo, _ = make_box_obb(T_base_box)
    table_obs          = make_table_slab(T_base_table, box_lo[2])

    static_obs = _default_obstacles()                  # torso
    obs_full   = static_obs + [box_obs, table_obs]     # approach: box present
    obs_no_box = static_obs +            [table_obs]   # grasp/lift: box dropped

    return Scene(T_base_box=T_base_box, T_base_table=T_base_table,
                 targets=targets, R_yaw=R_yaw,
                 box_obs=box_obs, table_obs=table_obs,
                 obs_full=obs_full, obs_no_box=obs_no_box)


# === IK validation ==========================================================

def solve_and_validate(arm, t_l, t_r, q_seed, obstacles, label,
                       obstacle_names=None, r_l=None, r_r=None):
    """14-DOF IK to (t_l, t_r) seeded at q_seed, then _validate the result.

    Returns the joint vector if the goal is reachable, in-limits and
    collision-free in `obstacles`; otherwise prints the reason (and, on a
    collision, the named colliding pairs via collision_report) and returns None.
    """
    arm.obstacles = np.array(obstacles, dtype=float)
    q = arm.inverse_kinematics(
        t_l, t_r, q_seed,
        target_left_ori=r_l,
        target_right_ori=r_r,
        orientation_weight=ORIENTATION_WEIGHT,
    )
    ok, reason = _validate(
        arm, q, target_a=t_l, target_b=t_r,
        target_a_ori=r_l, target_b_ori=r_r,
    )
    print(f"  {label} IK: {reason}")
    if not ok and "collision" in reason:
        for cap, obs in arm.collision_report(q, obstacle_names):
            print(f"      collide: {cap}  vs  {obs}")
    return q if ok else None


def report_ee(arm, q, target_l, target_r, label):
    """Print each hand's actual EE position vs its target and the error (mm).

    Print-only diagnostics; does not gate anything. Used after a leg finishes to
    eyeball how close the arm landed to the commanded grasp/pre-grasp point.
    """
    ee_l = arm.left.get_end_effector_position(q[:7])
    ee_r = arm.right.get_end_effector_position(q[7:])
    dl = ee_l - target_l
    dr = ee_r - target_r
    print(f"\n  --- {label} EE positions ---")
    print(f"    LEFT  target {target_l}")
    print(f"          actual {ee_l}   err={np.linalg.norm(dl) * 1000:5.1f} mm  "
          f"dxyz=({dl[0] * 1000:+.1f}, {dl[1] * 1000:+.1f}, {dl[2] * 1000:+.1f}) mm")
    print(f"    RIGHT target {target_r}")
    print(f"          actual {ee_r}   err={np.linalg.norm(dr) * 1000:5.1f} mm  "
          f"dxyz=({dr[0] * 1000:+.1f}, {dr[1] * 1000:+.1f}, {dr[2] * 1000:+.1f}) mm")


# === post-run visualization =================================================

def render_planned(arm, record, prefix):
    """Write one plotly HTML per multi-waypoint group AFTER a run — no extra
    planning, just renders the path run_sequence already streamed.

    Renders any group with >2 waypoints (RRT legs, and multi-sub-step Cartesian
    moves); trivial 2-point direct moves are skipped. `record` is the list filled
    by run_sequence(..., record=record). Restores the baseline (torso-only)
    obstacles on the model(s) before returning so the next run starts clean.
    """
    written = []
    for gi, g in enumerate(record):
        if len(g["path"]) <= 2:
            continue
        side = g["side"]
        model = (arm if side == "both"
                 else arm.left if side == "left" else arm.right)
        model.obstacles = np.array(g["obstacles"], dtype=float)
        safe = (str(g.get("label", f"g{gi}"))
                .replace(" → ", "_").replace(" ", "_")
                .replace("/", "_").replace("->", "to")
                .replace("(", "").replace(")", "").replace(":", ""))[:50]
        fn = f"{prefix}_{gi:02d}_{safe}.html"
        if save_path_html(model, g["path"], fn):
            written.append(fn)
    base = np.array(_default_obstacles(), dtype=float)
    arm.obstacles = base
    if hasattr(arm, "left"):  arm.left.obstacles  = base
    if hasattr(arm, "right"): arm.right.obstacles = base
    if written:
        print(f"[viz] wrote {len(written)} plan HTML(s): {', '.join(written)}")
    else:
        print(f"[viz] {prefix}: no RRT legs to render (all moves were direct).")