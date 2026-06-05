#!/usr/bin/env python3
"""
g1_plan_run.py

Cleaner end-to-end bridge: define obstacle(s) and a Cartesian target,
plan an RRT-Connect path in simulation, visualize it in matplotlib,
then (with confirmation) execute it on real hardware.

Workflow:
  - Edit the OBSTACLES and TARGET section at the bottom of this file
    (or pass --target X Y Z on the command line)
  - Run with --simonly to plan + visualize without ever touching the robot
  - Run normally to plan + visualize + ask to execute on hardware

Recommended order:
  1. python3 g1_plan_run.py --simonly        # confirm path looks sane
  2. python3 g1_plan_run.py                  # plan, visualize, confirm, run
"""

import argparse
import math
import time
import numpy as np

from myArm_v2 import G1Arm, rpy_to_R, _orientation_error
from rrt_planner import RRTConnect
from g1_arm_controller import G1ArmController

from g1_robot import G1Robot
from g1_hand_controller import GESTURES_RAD

from g1_choreograph    import run_sequence, preflight_validate
from choreographies    import CHOREOGRAPHIES

def _ori_desc(R):
    """Short label for an orientation target, for log output."""
    if R is None:
        return "None (position-only)"
    if np.allclose(R, np.eye(3), atol=1e-6):
        return "identity (palm +X torso)"
    return "custom"

def _apply_gestures(robot, args, label):
    """Fire the requested gestures and wait long enough for the slew to finish.
    `label` is just for the log line so we can tell 'before' from 'after'."""
    if not (args.left_gesture or args.right_gesture):
        return
    print(f"\nApplying gestures ({label})...")
    if args.left_gesture and robot.left_hand is not None:
        robot.left_hand.set_gesture(args.left_gesture)
    if args.right_gesture and robot.right_hand is not None:
        robot.right_hand.set_gesture(args.right_gesture)

    time.sleep(2.0)

def _default_obstacles():
    return [
        [0.0,   0.0,    0.05,    0, 0, 0, 0.12, 0.14, 0.50],   # torso
    ]
# The new G1Arm class loads the URDF directly and already extends the
# chain 12 cm past the palm so the last collision capsule covers the
# dex3 hand naturally. No manual hand-capsule fudge needed anymore.


def make_arm(obstacles_list):
    """Build a G1Arm with our obstacle list."""
    arm = G1Arm(side="left")
    if obstacles_list is None or len(obstacles_list) == 0:
        arm.obstacles = np.empty((0, 9))
    else:
        arm.obstacles = np.array(obstacles_list, dtype=float)
    return arm


def validate_goal(arm, q_goal, target_xyz, ik_tol_m=0.03, ik_max_m=0.10):
    """Return (ok, reason) for a candidate goal config."""
    if not arm.is_within_joint_limits(q_goal):
        return False, "goal outside joint limits"
    fk = arm.get_end_effector_position(q_goal)
    err = float(np.linalg.norm(np.array(target_xyz) - fk))
    if err > ik_max_m:
        return False, f"IK error {err*1000:.0f} mm > {ik_max_m*1000:.0f} mm"
    if arm.check_collision(q_goal):
        return False, "goal in collision with obstacles"
    return True, f"IK ok (err={err*1000:.1f} mm)"


def save_path_html(arm, path, filename, n_snapshots=8, target=None):
    """Render an interactive plotly HTML viewer of the planned path.
    Works headlessly. Open the file in any browser to rotate/zoom/inspect.
    Returns True on success, False if plotly isn't installed."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[viz] plotly not installed. Install with:")
        print("      python3 -m pip install plotly")
        return False

    fig = go.Figure()

    # Obstacles -> box wireframes
    obstacle_colors = ["red", "darkred", "firebrick", "crimson", "indianred"]
    for i, obs in enumerate(arm.obstacles):
        center = obs[:3]
        size = obs[6:9]
        r = size / 2
        Rb = rpy_to_R(obs[3], obs[4], obs[5])      # honor OBB orientation
        v = np.array([
            center + Rb @ np.array([sx * r[0], sy * r[1], sz * r[2]])
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
        ])
        edges = [(0, 1), (2, 3), (4, 5), (6, 7),
                 (0, 2), (1, 3), (4, 6), (5, 7),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        xs, ys, zs = [], [], []
        for a, b in edges:
            xs.extend([v[a][0], v[b][0], None])
            ys.extend([v[a][1], v[b][1], None])
            zs.extend([v[a][2], v[b][2], None])
        c = obstacle_colors[i % len(obstacle_colors)]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=c, width=4),
            name=f"obstacle {i}"))

    # Reference points: torso origin, left shoulder
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers+text",
        marker=dict(size=4, color="black"),
        text=["torso"], textposition="top center", name="torso"))
    fig.add_trace(go.Scatter3d(
        x=[0.004], y=[0.100], z=[0.238], mode="markers+text",
        marker=dict(size=5, color="black", symbol="square"),
        text=["L shoulder"], textposition="top center", name="L shoulder"))
    if hasattr(arm, "left") and hasattr(arm, "right"):
        fig.add_trace(go.Scatter3d(
            x=[0.004], y=[-0.100], z=[0.238], mode="markers+text",
            marker=dict(size=5, color="black", symbol="square"),
            text=["R shoulder"], textposition="top center", name="R shoulder"))

    # Arm snapshots in viridis-ish colors.
    # Dual-arm: left and right are drawn as separate lines.
    # Single-arm: existing behavior.
    is_dual = hasattr(arm, "left") and hasattr(arm, "right")
    FINGERTIP = 9   # fingertip link index in each 11-link arm chain (10=palm_under)
    if is_dual:
        n_left = len(arm.left.arm.links)
        left_idxs = list(arm.left.active_joint_indices) + [FINGERTIP]
        right_idxs = [i + n_left for i in arm.right.active_joint_indices] \
                     + [n_left + FINGERTIP]
    else:
        single_idxs = list(arm.active_joint_indices) + [FINGERTIP]

    n = min(n_snapshots, len(path))
    idxs = np.linspace(0, len(path) - 1, n).astype(int)
    for k, i in enumerate(idxs):
        all_p = np.array(arm.get_link_position(path[i]))
        t = k / max(1, n - 1)
        r = int(68 + t * 180); g = int(1 + t * 220); b = int(84 + t * 100)
        color = f"rgb({r},{min(g,255)},{b})"

        if is_dual:
            for tag, sel in (("L", left_idxs), ("R", right_idxs)):
                jp = all_p[sel]
                fig.add_trace(go.Scatter3d(
                    x=jp[:, 0], y=jp[:, 1], z=jp[:, 2],
                    mode="lines+markers",
                    line=dict(width=6, color=color),
                    marker=dict(size=4, color=color),
                    name=f"{tag} wp {i}",
                    showlegend=(k in (0, n - 1))))
        else:
            jp = all_p[single_idxs]
            fig.add_trace(go.Scatter3d(
                x=jp[:, 0], y=jp[:, 1], z=jp[:, 2],
                mode="lines+markers",
                line=dict(width=6, color=color),
                marker=dict(size=4, color=color),
                name=f"arm wp {i}",
                showlegend=(k in (0, n - 1))))

    # EE trajectory + start/goal markers (per-arm in dual mode)
    if is_dual:
        ee_l = np.array([arm.left.get_end_effector_position(q[:7]) for q in path])
        ee_r = np.array([arm.right.get_end_effector_position(q[7:]) for q in path])
        for ee_arr, tag, c in [(ee_l, "L", "royalblue"), (ee_r, "R", "darkorange")]:
            fig.add_trace(go.Scatter3d(
                x=ee_arr[:, 0], y=ee_arr[:, 1], z=ee_arr[:, 2],
                mode="lines", line=dict(width=4, color=c),
                name=f"{tag} EE path"))
            fig.add_trace(go.Scatter3d(
                x=[ee_arr[0, 0]], y=[ee_arr[0, 1]], z=[ee_arr[0, 2]],
                mode="markers", marker=dict(size=8, color="green"),
                name=f"{tag} start", showlegend=False))
            fig.add_trace(go.Scatter3d(
                x=[ee_arr[-1, 0]], y=[ee_arr[-1, 1]], z=[ee_arr[-1, 2]],
                mode="markers", marker=dict(size=10, color="red",
                                            symbol="diamond"),
                name=f"{tag} goal", showlegend=False))
    else:
        ee = np.array([arm.get_end_effector_position(q) for q in path])
        fig.add_trace(go.Scatter3d(
            x=ee[:, 0], y=ee[:, 1], z=ee[:, 2],
            mode="lines", line=dict(width=4, color="royalblue"),
            name="EE path"))
        fig.add_trace(go.Scatter3d(
            x=[ee[0, 0]], y=[ee[0, 1]], z=[ee[0, 2]],
            mode="markers", marker=dict(size=8, color="green"),
            name="Start"))
        fig.add_trace(go.Scatter3d(
            x=[ee[-1, 0]], y=[ee[-1, 1]], z=[ee[-1, 2]],
            mode="markers", marker=dict(size=10, color="red", symbol="diamond"),
            name="Goal (achieved)"))
    if target is not None:
        fig.add_trace(go.Scatter3d(
            x=[target[0]], y=[target[1]], z=[target[2]],
            mode="markers", marker=dict(size=7, color="orange", symbol="x"),
            name="Target (requested)"))

    fig.update_layout(
        title=f"G1 path: {len(path)} waypoints, "
              f"{n} snapshots, {len(arm.obstacles)} obstacles",
        scene=dict(
            xaxis_title="X (forward)",
            yaxis_title="Y (left)",
            zaxis_title="Z (up)",
            aspectmode="data"))
    fig.write_html(filename)
    print(f"Saved interactive HTML to: {filename}")
    return True


def save_path_png(arm, path, filename, n_snapshots=6):
    """Render a static multi-pose snapshot of the planned path with
    obstacles and EE trajectory. Works headlessly (no X display needed).
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive; no display required
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Obstacles as red wireframe boxes
    for obs in arm.obstacles:
        center = np.asarray(obs[:3], float)
        size = obs[6:9]
        r = size / 2
        Rb = rpy_to_R(obs[3], obs[4], obs[5])      # honor OBB orientation
        v = [
            center + Rb @ np.array([sx * r[0], sy * r[1], sz * r[2]])
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
        ]
        edges = [(0, 1), (1, 3), (3, 2), (2, 0),
                 (4, 5), (5, 7), (7, 6), (6, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        for a, b in edges:
            ax.plot3D(*zip(v[a], v[b]), "r-", alpha=0.6, linewidth=1)

    # Arm snapshots, fading from light->dark across the path.
    # Dual-arm: draw left and right separately so they don't link visually.
    is_dual = hasattr(arm, "left") and hasattr(arm, "right")
    FINGERTIP = 9   # fingertip link index in each 11-link arm chain (10=palm_under)
    if is_dual:
        n_left = len(arm.left.arm.links)
        left_idxs = list(arm.left.active_joint_indices) + [FINGERTIP]
        right_idxs = [i + n_left for i in arm.right.active_joint_indices] \
                     + [n_left + FINGERTIP]
    else:
        single_idxs = list(arm.active_joint_indices) + [FINGERTIP]
    cmap = plt.cm.viridis
    n_snapshots = min(n_snapshots, len(path))
    idxs = np.linspace(0, len(path) - 1, n_snapshots).astype(int)
    for k, i in enumerate(idxs):
        all_positions = np.array(arm.get_link_position(path[i]))
        c = cmap(k / max(1, n_snapshots - 1))
        if is_dual:
            for sel in (left_idxs, right_idxs):
                joint_pos = all_positions[sel]
                ax.plot(joint_pos[:, 0], joint_pos[:, 1], joint_pos[:, 2],
                        color=c, alpha=0.75, linewidth=2.5)
                ax.scatter(joint_pos[:, 0], joint_pos[:, 1], joint_pos[:, 2],
                           color=c, s=25, alpha=0.75)
        else:
            joint_pos = all_positions[single_idxs]
            ax.plot(joint_pos[:, 0], joint_pos[:, 1], joint_pos[:, 2],
                    color=c, alpha=0.75, linewidth=2.5,
                    label=f"wp {i}" if k in (0, n_snapshots - 1) else None)
            ax.scatter(joint_pos[:, 0], joint_pos[:, 1], joint_pos[:, 2],
                       color=c, s=25, alpha=0.75)

    # EE trajectory (per-arm in dual mode)
    if is_dual:
        for getter, c in [(arm.left.get_end_effector_position, "b-"),
                          (arm.right.get_end_effector_position, "darkorange")]:
            ee_pts = np.array([
                getter(q[:7] if c == "b-" else q[7:]) for q in path])
            ax.plot(ee_pts[:, 0], ee_pts[:, 1], ee_pts[:, 2],
                    c, linewidth=1.2, alpha=0.6)
            ax.scatter(*ee_pts[0], color="green", s=120, marker="o")
            ax.scatter(*ee_pts[-1], color="red", s=140, marker="*")
    else:
        ee_pts = np.array([arm.get_end_effector_position(q) for q in path])
        ax.plot(ee_pts[:, 0], ee_pts[:, 1], ee_pts[:, 2],
                "b-", linewidth=1.2, alpha=0.6, label="EE path")
        ax.scatter(*ee_pts[0], color="green", s=180, marker="o", label="Start")
        ax.scatter(*ee_pts[-1], color="red", s=200, marker="*", label="Goal")

    # Reference markers: torso origin (0,0,0) and the left shoulder so you
    # can orient yourself in the plot.
    ax.scatter(0, 0, 0, color="black", s=60, marker="x")
    ax.text(0, 0, 0, "  torso", color="black", fontsize=8)
    shoulder = np.array([0.0040, 0.10022, 0.23778])
    ax.scatter(*shoulder, color="black", s=40, marker="s")
    ax.text(*shoulder, "  L shoulder", color="black", fontsize=8)
    if is_dual:
        rshoulder = np.array([0.0040, -0.10022, 0.23778])
        ax.scatter(*rshoulder, color="black", s=40, marker="s")
        ax.text(*rshoulder, "  R shoulder", color="black", fontsize=8)

    ax.set_xlabel("X (forward)")
    ax.set_ylabel("Y (left)")
    ax.set_zlabel("Z (up)")
    ax.set_title(f"Planned path: {len(path)} waypoints, "
                 f"{n_snapshots} arm snapshots, {len(arm.obstacles)} obstacles")
    ax.legend(loc="upper left")

    # Reasonable view + bounds
    all_pts = np.vstack([ee_pts, *[obs[:3] for obs in arm.obstacles]]) \
        if len(arm.obstacles) else ee_pts
    pad = 0.2
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    ax.set_zlim(all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad)

    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved path snapshot to: {filename}")


def visualize(arm, path, save_png=None, save_html=None, live=False, target=None):
    """Save outputs in any/all of: PNG snapshot, plotly HTML, live animation."""
    if save_html:
        save_path_html(arm, path, save_html, target=target)
    if save_png:
        save_path_png(arm, path, save_png)
    if live:
        print("Showing matplotlib animation. Close the window to continue.")
        arm.visualize_path(path, obstacles=True, pause_time=0.1)


def plan(arm, q_start, q_goal, step_size=0.15):
    print(f"Planning RRT-Connect (step={step_size})...")
    t_total_start = time.perf_counter()

    t_rrt_start = time.perf_counter()
    planner = RRTConnect(arm, step_size=step_size)
    path = planner.plan(q_start, q_goal)
    t_rrt = time.perf_counter() - t_rrt_start

    if not path:
        print(f"[plan] RRT failed in {t_rrt:.2f}s — no path.")
        return path

    t_sc_start = time.perf_counter()
    shortened = shortcut_path(arm, path)
    t_sc = time.perf_counter() - t_sc_start

    t_total = time.perf_counter() - t_total_start
    print(f"[plan] TOTAL {t_total:.2f}s  "
          f"(RRT {t_rrt:.2f}s [{100*t_rrt/t_total:.0f}%], "
          f"shortcut {t_sc:.2f}s [{100*t_sc/t_total:.0f}%])")
    return shortened


def _bisection_order(n):
    """Yield n distinct sample points in (0, 1) in bisection-depth order:
    midpoint first, then quarter-points, then eighth-points, etc.

    Why this order: along a straight-line shortcut between two waypoints
    that are already verified collision-free, any collision is most likely
    to occur near the *middle* of the segment (the point furthest from
    both verified endpoints). With linear sampling, the middle is sample
    7/15; we waste 7 collision checks before finding the failure. With
    bisection ordering, the middle is sample 1, and the average failure
    is detected within the first 2-3 samples.

    For n=15, yields: 0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875,
    1/16, 3/16, 5/16, 7/16, 9/16, 11/16, 13/16, 15/16.
    Same coverage as linspace(0,1,17)[1:-1], better failure-detection
    ordering, and natively excludes the (already-verified) endpoints.
    """
    if n <= 0:
        return
    yield 0.5
    count = 1
    depth = 2
    while count < n:
        denom = 1 << depth
        for num in range(1, denom, 2):
            yield num / denom
            count += 1
            if count >= n:
                return
        depth += 1


def shortcut_path(arm, path, max_passes=10, resolution_rad=0.10,
                  random_stop_after=3, verbose=True):
    """Post-process: shorten the path by replacing subpaths with direct
    linear interpolations when those are collision-free.

    Two passes:
      1. Deterministic forward greedy: for each anchor i (left to right),
         try the longest possible shortcut to j = N-1, N-2, ... Take the
         first j that gives a collision-free linear segment. This
         guarantees no obvious shortcut is missed.
      2. Random polish: max_passes random (i, j) attempts on top, in case
         re-ordering enables further shortcuts (rare but happens when
         shortcut B only works after shortcut A has fired).

    Sample count per segment is *adaptive*: each shortcut is sampled
    densely enough that the max joint-space gap between samples stays
    under resolution_rad. Short shortcuts get 3-5 samples, long ones can
    get 20+. Sampling order is bisection-first (midpoint, then quarters,
    etc.), so collisions in the middle of a shortcut are detected in the
    first few samples rather than after a full linear scan. Endpoints
    are skipped — they came from a verified path and don't need rechecking.
    """
    if len(path) < 3:
        return path
    p = [np.asarray(q, dtype=float) for q in path]
    is_dual = hasattr(arm, "left") and hasattr(arm, "right")

    # Tolerance for joint-limit checks. Pulled from the arm class so we
    # match whatever the upstream methods use. ~1e-9 rad absorbs FP drift
    # (especially through locked / zero-width joints) without masking any
    # physically meaningful violation.
    _arm_for_tol = arm.left if is_dual else arm
    JL_TOL = getattr(type(_arm_for_tol), "JOINT_LIMIT_TOL", 1e-9)

    # ---- sanity check: are the input waypoints themselves within limits?
    # Joint limits form a convex set, so interpolations between valid
    # waypoints should always be valid. If we see joint-limit failures
    # during shortcutting, the likely culprits are: (a) some waypoint
    # already exceeds limits, or (b) floating-point drift on a locked
    # (zero-width) joint. Print both diagnostics up front.
    if verbose:
        violators = []
        for wp_idx, q in enumerate(p):
            for j in range(len(q)):
                lo = arm.joint_limits[j, 0]
                hi = arm.joint_limits[j, 1]
                if q[j] < lo - JL_TOL or q[j] > hi + JL_TOL:
                    delta = max(lo - q[j], q[j] - hi)
                    violators.append((wp_idx, j, q[j], lo, hi, delta))
        if violators:
            print(f"[shortcut] WARNING: {len(violators)} waypoint-joint pairs "
                  f"already exceed limits in the input path:")
            for wp_idx, j, val, lo, hi, delta in violators[:5]:
                print(f"[shortcut]   wp {wp_idx} joint {j}: {val:.9f} "
                      f"outside [{lo:.9f}, {hi:.9f}] by {delta:.2e}")
            if len(violators) > 5:
                print(f"[shortcut]   ... and {len(violators) - 5} more")
        zero_width = [j for j in range(arm.joint_limits.shape[0])
                      if arm.joint_limits[j, 1] - arm.joint_limits[j, 0] < 1e-12]
        if zero_width:
            print(f"[shortcut] Note: zero-width joint limits on indices "
                  f"{zero_width} (locked joints; tolerance {JL_TOL:.0e} "
                  f"absorbs FP drift)")

    # Item 2: hoist invariant work out of the inner substep loop.
    # The obstacle set does not change during a shortcut call, so push it to
    # the per-arm models ONCE here instead of re-assigning on every substep
    # of every candidate segment (previously thousands of redundant writes).
    if is_dual:
        arm.left.obstacles = arm.obstacles
        arm.right.obstacles = arm.obstacles
    # Cache the joint-limit bounds as flat arrays so the per-substep check is
    # a vectorized np.any instead of a 14-iteration Python loop.
    _jl_lo = arm.joint_limits[:, 0]
    _jl_hi = arm.joint_limits[:, 1]

    # Stats
    stats = {
        "checks": 0,
        "cleared": 0,
        "fail_joint_limit": 0,
        "fail_collision": 0,
        "fail_left_obstacle": 0,
        "fail_right_obstacle": 0,
        "fail_inter_arm": 0,
    }
    # Track worst joint-limit violation across all failed segments
    worst_jl = {"delta": 0.0, "joint": None, "val": None, "lo": None,
                "hi": None, "i": None, "j": None}
    # Histogram: which joint is the most common offender?
    jl_joint_hits = {}

    def segment_status(qa, qb, i_idx=None, j_idx=None):
        # Adaptive sample count: ensure max joint-space gap between samples
        # stays <= resolution_rad. Short shortcuts get few samples (the
        # work pays for itself only on long ones where a clip is plausible).
        max_delta_rad = float(np.max(np.abs(qb - qa)))
        # +1 because bisection order over n samples in (0,1) has max gap
        # 1/(n+1). Cap at 50 — defensive against pathological inputs.
        n_samples = max(3, int(math.ceil(max_delta_rad / resolution_rad)))
        n_samples = min(n_samples, 50)
        for s in _bisection_order(n_samples):
            q = (1 - s) * qa + s * qb
            # Vectorized joint-limit check (item 2): one np.any pair instead
            # of a Python loop over all 14 joints per substep. Only when a
            # violation is found do we drop into the (rare) per-joint path to
            # populate the diagnostic histogram / worst-offender record.
            if np.any(q < _jl_lo - JL_TOL) or np.any(q > _jl_hi + JL_TOL):
                below = _jl_lo - q
                above = q - _jl_hi
                jj = int(np.argmax(np.maximum(below, above)))
                lo = _jl_lo[jj]
                hi = _jl_hi[jj]
                delta = max(lo - q[jj], q[jj] - hi)
                jl_joint_hits[jj] = jl_joint_hits.get(jj, 0) + 1
                if delta > worst_jl["delta"]:
                    worst_jl.update(delta=delta, joint=jj, val=q[jj],
                                    lo=lo, hi=hi, i=i_idx, j=j_idx)
                return "joint_limit"
            if is_dual:
                ql, qr = q[:7], q[7:]
                # obstacles already set on arm.left / arm.right once, above.
                if arm.left.check_collision(ql):
                    return "left_obstacle"
                if arm.right.check_collision(qr):
                    return "right_obstacle"
                if arm._check_inter_arm(ql, qr):
                    return "inter_arm"
            else:
                if arm.check_collision(q):
                    return "collision"
        return "clear"

    def segment_clear(qa, qb, i_idx=None, j_idx=None):
        stats["checks"] += 1
        status = segment_status(qa, qb, i_idx, j_idx)
        if status == "clear":
            stats["cleared"] += 1
            return True
        stats[f"fail_{status}"] += 1
        return False

    initial_len = len(p)

    # Pass 1: deterministic forward greedy
    t_pass1 = time.perf_counter()
    i = 0
    pass1_dropped = 0
    while i < len(p) - 2:
        for j in range(len(p) - 1, i + 1, -1):
            if segment_clear(p[i], p[j], i_idx=i, j_idx=j):
                pass1_dropped += (j - i - 1)
                p = p[:i + 1] + p[j:]
                break
        i += 1
    len_after_pass1 = len(p)
    t_pass1 = time.perf_counter() - t_pass1

    # Pass 2: random polish, with early-stop. Profiling showed the greedy
    # pass already reaches the final waypoint count on realistic scenes and
    # the random pass almost never finds an additional shortcut, so the
    # budget is small and we bail after `random_stop_after` consecutive
    # attempts that drop nothing. This keeps the (rare) benefit while
    # removing the wasted seg checks that dominated shortcut time on the
    # robot. Set random_stop_after=None to disable early-stop.
    t_pass2 = time.perf_counter()
    pass2_dropped = 0
    pass2_used = 0
    no_improve = 0
    for _ in range(max_passes):
        if len(p) < 3:
            break
        pass2_used += 1
        i = np.random.randint(0, len(p) - 2)
        j = np.random.randint(i + 2, len(p))
        if segment_clear(p[i], p[j], i_idx=i, j_idx=j):
            pass2_dropped += (j - i - 1)
            p = p[:i + 1] + p[j:]
            no_improve = 0
        else:
            no_improve += 1
            if random_stop_after is not None and no_improve >= random_stop_after:
                break
    t_pass2 = time.perf_counter() - t_pass2

    if verbose:
        print(f"[shortcut] Input: {initial_len} waypoints")
        print(f"[shortcut] After pass 1 (forward greedy): "
              f"{len_after_pass1} wp (dropped {pass1_dropped}) "
              f"in {t_pass1:.2f}s")
        print(f"[shortcut] After pass 2 (random polish):  "
              f"{len(p)} wp (dropped {pass2_dropped} more) "
              f"in {t_pass2:.2f}s [{pass2_used}/{max_passes} passes used]")
        total = stats["checks"]
        cleared = stats["cleared"]
        blocked = total - cleared
        print(f"[shortcut] segment_clear calls: {total} total, "
              f"{cleared} cleared ({100*cleared/max(total,1):.0f}%), "
              f"{blocked} blocked")
        if blocked > 0:
            print(f"[shortcut] Failure breakdown:")
            categories = [
                ("fail_joint_limit", "joint limits"),
                ("fail_collision", "collision (single arm)"),
                ("fail_left_obstacle", "left arm vs obstacle"),
                ("fail_right_obstacle", "right arm vs obstacle"),
                ("fail_inter_arm", "inter-arm collision"),
            ]
            for key, label in categories:
                count = stats[key]
                if count > 0:
                    pct = 100 * count / blocked
                    print(f"[shortcut]   {label:30s}: {count:5d} "
                          f"({pct:5.1f}% of failures)")
        if stats["fail_joint_limit"] > 0:
            print(f"[shortcut] Joint-limit failure detail:")
            # Which joints get hit the most
            top_joints = sorted(jl_joint_hits.items(),
                                key=lambda kv: -kv[1])[:3]
            for jj, hits in top_joints:
                lo = arm.joint_limits[jj, 0]
                hi = arm.joint_limits[jj, 1]
                width = hi - lo
                print(f"[shortcut]   joint {jj}: {hits} hits, "
                      f"limits=[{lo:.6f}, {hi:.6f}] (width={width:.2e})")
            # Worst single violation
            w = worst_jl
            print(f"[shortcut]   worst case: joint {w['joint']} value "
                  f"{w['val']:.9f} outside [{w['lo']:.9f}, {w['hi']:.9f}] "
                  f"by {w['delta']:.2e} (during shortcut try i={w['i']}, "
                  f"j={w['j']})")

    return p


def confirm(prompt):
    return input(f"{prompt} [type 'yes' to proceed]: ").strip().lower() == "yes"


def main():
    _GESTURE_CHOICES = sorted(GESTURES_RAD["left"].keys())
    ap = argparse.ArgumentParser()
    ap.add_argument("--simonly", action="store_true",
                    help="Plan + visualize only; never connect to hardware")
    ap.add_argument("--target", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    default=None,
                    help="Single-arm target XYZ (only used when --side is left or right)")
    ap.add_argument("--target-left", nargs=3, type=float,
                    metavar=("X", "Y", "Z"), default=None,
                    help="Left-arm target XYZ for --side both")
    ap.add_argument("--target-right", nargs=3, type=float,
                    metavar=("X", "Y", "Z"), default=None,
                    help="Right-arm target XYZ for --side both")
    ap.add_argument("--side", default="left",
                    choices=("left", "right", "both"))
    ap.add_argument("--step", type=float, default=0.15)
    ap.add_argument("--save-html", default="/tmp/g1_path.html",
                    metavar="PATH",
                    help="Save interactive plotly HTML (default: /tmp/g1_path.html)")
    ap.add_argument("--no-viz", action="store_true",
                    help="Skip visualization entirely")
    ap.add_argument("--save-viz", default="/tmp/g1_path.png",
                    metavar="PATH",
                    help="Save static path snapshot PNG (default: /tmp/g1_path.png)")
    ap.add_argument("--live-viz", action="store_true",
                    help="Try live matplotlib animation (needs X display)")
    ap.add_argument("--left-gesture", default=None, choices=_GESTURE_CHOICES, help="left gesture control")
    ap.add_argument("--right-gesture", default=None, choices=_GESTURE_CHOICES, help="right gesture control")
    ap.add_argument("--gesture-when", default="after", choices=("before", "after"),help="gesture before/after hand move")
    ap.add_argument("--choreography", default=None,
                choices=sorted(CHOREOGRAPHIES.keys()),
                help="Run a named choreography instead of a single target. "
                     "Ignores --target, --side, --target-left/right, and "
                     "--left/right-gesture (steps define their own gestures).")
    ap.add_argument("--gravity-comp-scale", type=float, default=1.0,
                help="Gravity-compensation feedforward scale, 0..1. "
                     "0 (default) = no comp, same as before. Start small "
                     "(0.3, 0.5) and ramp up while observing for oscillation. "
                     "1.0 = full URDF-computed torque applied as tau_ff.")
    ap.add_argument("--gravity-comp-sign", type=float, default=+1.0,
                help="Sign of gravity comp torque (+1 or -1). Flip to -1 if "
                     "hardware testing shows sag getting worse with comp on "
                     "(indicates arm_sdk firmware convention is opposite).")
    args = ap.parse_args()

    # ---------- choreography branch ----------
    if args.choreography:
        if args.simonly:
            print("[choreograph] --simonly not yet supported for choreographies. "
                "Aborting.")
            return

        # Reuse the OBSTACLES list defined below by reading it from the closure.
        # We have to define it inline here since the existing OBSTACLES is below
        # this block — pull it up or just duplicate the minimum for the test.
        
        steps = CHOREOGRAPHIES[args.choreography]

        # Choose arm model based on whether the sequence has any dual steps.
        # For dual we defer the G1DualArm build until inside the G1Robot
        # context so we can read the broken right-wrist-yaw encoder and lock
        # that joint in the planner. Same pattern as the dual single-target
        # path below.
        needs_dual = any(s.get("side") == "both" for s in steps)
        if needs_dual:
            from myArm_dual import G1DualArm
            arm_model = None     # built after connect()
            model_desc = "G1DualArm (14 DOF) — RWY lock pending hardware read"
            # Temporary arm_model used only for preflight validation,
            # discarded before runtime. No locked_joints (RWY angle is
            # unknown until hardware connect); see preflight_validate
            # docstring for what that implies.
            preflight_arm = G1DualArm()
            preflight_arm.obstacles = np.array(_default_obstacles(),
                                               dtype=float)
        else:
            arm_model = make_arm(_default_obstacles())
            model_desc = "G1Arm (single, 7 DOF)"
            preflight_arm = arm_model     # reuse the runtime model

        print(f"\nChoreography     : {args.choreography} ({len(steps)} steps)")
        print(f"Arm model        : {model_desc}")
        for i, s in enumerate(steps):
            print(f"  {i+1:2d}. {s.get('label', '(unlabeled)'):30s} "
                f"side={s['side']:5s}  "
                f"{'joint-space' if 'target_q' in s else 'task-space '}"
                f"{' [skip plan]' if s.get('skip_planning') else ''}")

        # Pre-flight validation — catches obvious goal errors before the
        # user commits to a hardware run. Locked-joint specifics are
        # verified at runtime since the locked angle isn't known yet.
        ok, _issues = preflight_validate(preflight_arm, steps)
        if not ok:
            print("\nAborting before hardware connect "
                  "(preflight validation failed).")
            return

        if not confirm(f"\nRun choreography '{args.choreography}' "
                    f"on REAL HARDWARE?"):
            print("Aborted.")
            return

        # Auto-detect which hands are needed from the gesture references
        def _hand_used(sides_wanted):
            for s in steps:
                for hook in ("gestures_before", "gestures_after"):
                    g = s.get(hook) or {}
                    for side in sides_wanted:
                        if g.get(side):
                            return True
            return False

        needs_left  = _hand_used(["left"])
        needs_right = _hand_used(["right"])
        print(f"Hands required   : left={needs_left}  right={needs_right}")

        # Optional gravity-compensation feedforward. Loads pinocchio model
        # from the URDF; arm joints only by default (waist excluded for
        # safety — see GravityComp docstring).
        gc = None
        if args.gravity_comp_scale > 0.0:
            from gravity_comp import GravityComp
            gc = GravityComp(
                urdf_path="g1_29dof_with_hand.urdf",
                scale=args.gravity_comp_scale,
                sign=args.gravity_comp_sign,
            )

        with G1Robot(
            arm        = True,
            left_hand  = needs_left,
            right_hand = needs_right,
            max_arm_speed_dps = 80.0,
            kp_arm = 120.0,
            kd_arm = 2.0,
            gravity_comp = gc,
        ) as robot:
            # For dual sequences: lock right RWY at its current encoder
            # reading. The motor is broken so we can't move it; the planner
            # now holds q[13] constant so we never command motion that the
            # motor can't follow (which would otherwise diverge from the
            # plan and trip the chain_start drift safety check).
            # FK and collision-checking are unchanged.
            # >>> When RWY is fixed, delete this rebuild block. <<<
            if needs_dual:
                q_now = robot.arm.get_both_arms_q()
                current_rwy_right = q_now[13]
                print(f"Right RWY locked at {np.degrees(current_rwy_right):+6.2f} "
                      f"deg (motor broken)")
                arm_model = G1DualArm(
                    locked_joints_right={6: current_rwy_right})
                arm_model.obstacles = np.array(_default_obstacles(),
                                               dtype=float)
                # Tell the arm controller which joints are locked, so its
                # tracking-error abort skips them (they can't track by
                # definition; we still want the check on all other joints).
                robot.arm.locked_right = list(
                    arm_model.right.locked_joints.keys())

            run_sequence(robot, arm_model, steps)

        return
    # ---------- end choreography branch ----------

    # =========================================================
    #            EDIT THIS SECTION FOR YOUR SETUP
    # =========================================================

    # Obstacle list. Each row: [cx, cy, cz, 0, 0, 0, sx, sy, sz]
    OBSTACLES = _default_obstacles()

    # Default targets per mode
    TARGET_SINGLE = np.array([0.30, 0.30, 0.05])   # left arm default
    TARGET_LEFT   = np.array([0.50,  0.05, 0.25])  # both: left
    TARGET_RIGHT  = np.array([0.40, -0.05, 0.15])  # both: right

    # Orientation targets (3x3 rotation matrices in torso frame).
    # rpy_to_R(roll, pitch, yaw) uses ROS convention (intrinsic XYZ).
    # Examples:
    #   rpy_to_R(0, 0, 0)       -> identity: palm pointing +X (forward)
    #   rpy_to_R(0,  pi/2, 0)   -> palm pointing -Z (down)
    #   rpy_to_R(0, -pi/2, 0)   -> palm pointing +Z (up)
    #   rpy_to_R(0, 0,  pi/2)   -> palm pointing +Y (left)
    # To match the *current* palm orientation, capture FK at q_start.
    # Set to None to disable orientation control on that arm.
    TARGET_LEFT_ORI  = rpy_to_R(math.pi/2, 0, -math.pi/18)  # palm forward
    TARGET_RIGHT_ORI = None                     # RWY broken -> pos only
    # =========================================================

    dual = (args.side == "both")
    if dual:
        from myArm_dual import G1DualArm
        # Build without lock initially. In hardware mode we'll rebuild
        # with the current RWY reading after we connect to the controller.
        # In simonly mode there's no live encoder, so the right arm stays
        # unlocked (RRT samples RWY uniformly, same as before the lock).
        arm = G1DualArm()
        arm.obstacles = np.array(OBSTACLES, dtype=float)
        target_l = np.array(args.target_left) if args.target_left else TARGET_LEFT
        target_r = np.array(args.target_right) if args.target_right else TARGET_RIGHT
        print(f"Mode              : dual-arm (14 DOF)")
        print(f"Target left       : {target_l}")
        print(f"Target right      : {target_r}")
        print(f"Left  ori target  : {_ori_desc(TARGET_LEFT_ORI)}")
        print(f"Right ori target  : {_ori_desc(TARGET_RIGHT_ORI)}")
    else:
        arm = make_arm(OBSTACLES)
        target = np.array(args.target) if args.target else TARGET_SINGLE
        print(f"Mode              : single-arm ({args.side})")
        print(f"Target XYZ        : {target}")
    print(f"Obstacles         : {len(OBSTACLES)}")

    # ----- determine q_start -----
    if args.simonly:
        if dual:
            q_l = np.zeros(7); q_l[1] =  math.radians(20.0)
            q_r = np.zeros(7); q_r[1] = -math.radians(20.0)
            q_start = np.concatenate([q_l, q_r])
            print(f"\n[simonly] Synthetic start: both arms at clearance pose")
        else:
            q_start = np.zeros(7); q_start[1] = math.radians(20.0)
            print(f"\n[simonly] Synthetic start: clearance pose (LSR=20)")

        if arm.check_collision(q_start):
            print("[ABORT] Start pose collides with declared obstacles. "
                  "Shrink the body box. Saving diagnostic HTML.")
            if not args.no_viz and args.save_html:
                save_path_html(arm, [q_start, q_start], args.save_html)
            return

        # Compute goal
        if dual:
            q_goal = arm.inverse_kinematics(
                target_l, target_r, q_start,
                target_left_ori=TARGET_LEFT_ORI,
                target_right_ori=TARGET_RIGHT_ORI,
                orientation_weight=0.5)
        else:
            q_goal = arm.inverse_kinematics(target, initial_position=q_start)

        # Validate
        ok, reason = _validate(arm, q_goal,
                               target_l if dual else target,
                               target_r if dual else None,
                               target_a_ori=TARGET_LEFT_ORI if dual else None,
                               target_b_ori=TARGET_RIGHT_ORI if dual else None)
        print(f"Goal validation : {reason}")
        if not ok:
            print("[ABORT] " + reason)
            if not args.no_viz and args.save_html:
                save_path_html(arm, [q_start, q_goal], args.save_html)
            return

        path = plan(arm, q_start, q_goal, step_size=args.step)
        if not path:
            print("[ABORT] Planner found no path.")
            if not args.no_viz and args.save_html:
                save_path_html(arm, [q_start, q_goal], args.save_html)
            return
        print(f"Path: {len(path)} waypoints.")

        if not args.no_viz:
            visualize(arm, path,
                      save_png=args.save_viz or None,
                      save_html=args.save_html or None,
                      live=args.live_viz,
                      target=None if dual else target)
        return

    # ----- hardware run -----
    # Same gravity-comp flag as the choreography branch.
    gc = None
    if args.gravity_comp_scale > 0.0:
        from gravity_comp import GravityComp
        gc = GravityComp(
            urdf_path="g1_29dof_with_hand.urdf",
            scale=args.gravity_comp_scale,
            sign=args.gravity_comp_sign,
        )

    with G1Robot(
        arm        = True,
        left_hand  = (args.left_gesture is not None),
        right_hand = (args.right_gesture is not None),
        max_arm_speed_dps = 80.0,
        kp_arm= 120,
        kd_arm=2,
        gravity_comp = gc,
    ) as robot:
        ctrl = robot.arm  # convenience alias — keeps the rest of the block readable

        print("\nMoving to clearance pose...")
        if dual:
            ql = ctrl.get_current_q("left");  ql[1] =  math.radians(20.0)
            qr = ctrl.get_current_q("right"); qr[1] = -math.radians(20.0)
            ctrl.move_both_arms_to(np.concatenate([ql, qr]))
            time.sleep(0.5)
            q_start = ctrl.get_both_arms_q()

            # Lock right RWY at its current encoder reading. The motor is
            # broken so we can't move it; the planner now holds q[13]
            # constant so we never command motion that the motor can't
            # follow (which would otherwise diverge from the plan over
            # time and trip the joint-tracking safety check).
            # FK and collision-checking are identical to the unlocked
            # arm, so EE positions and visualization are unchanged.
            # >>> When RWY is fixed, delete this rebuild block. <<<
            current_rwy_right = q_start[13]
            print(f"Right RWY locked at {np.degrees(current_rwy_right):+6.2f} "
                  f"deg (motor broken)")
            arm = G1DualArm(locked_joints_right={6: current_rwy_right})
            arm.obstacles = np.array(OBSTACLES, dtype=float)

            ee_l_start = arm.left.get_end_effector_position(q_start[:7])
            ee_r_start = arm.right.get_end_effector_position(q_start[7:])
            print(f"q_start L (deg) : {np.round(np.degrees(q_start[:7]), 1)}")
            print(f"q_start R (deg) : {np.round(np.degrees(q_start[7:]), 1)}")
            print(f"EE_start L (m)  : {np.round(ee_l_start, 3)}")
            print(f"EE_start R (m)  : {np.round(ee_r_start, 3)}")
        else:
            q_clear = ctrl.get_current_q(args.side)
            q_clear[1] = math.radians(20.0)
            ctrl.move_arm_to(q_clear, side=args.side)
            time.sleep(0.5)
            q_start = ctrl.get_current_q(args.side)
            ee_start = arm.get_end_effector_position(q_start)
            print(f"q_start (deg)   : {np.round(np.degrees(q_start), 1)}")
            print(f"EE_start (m)    : {np.round(ee_start, 3)}")

        if arm.check_collision(q_start):
            print("[ABORT] Start in collision. Calibrate obstacles first.")
            if not args.no_viz and args.save_html:
                save_path_html(arm, [q_start, q_start], args.save_html)
            return

        # ---- solve IK ----
        if dual:
            q_goal = arm.inverse_kinematics(
                target_l, target_r, q_start,
                target_left_ori=TARGET_LEFT_ORI,
                target_right_ori=TARGET_RIGHT_ORI,
                orientation_weight=0.5)
        else:
            q_goal = arm.inverse_kinematics(target, initial_position=q_start)

        ok, reason = _validate(arm, q_goal,
                               target_l if dual else target,
                               target_r if dual else None,
                               target_a_ori=TARGET_LEFT_ORI if dual else None,
                               target_b_ori=TARGET_RIGHT_ORI if dual else None)
        print(f"\nGoal validation : {reason}")
        if not ok:
            print("[ABORT] " + reason)
            if not args.no_viz and args.save_html:
                save_path_html(arm, [q_start, q_goal], args.save_html)
            return

        print(f"q_goal (deg): {np.round(np.degrees(q_goal), 1)}")
        print(f"joint deltas (deg): "
              f"{np.round(np.degrees(q_goal - q_start), 1)}")

        path = plan(arm, q_start, q_goal, step_size=args.step)
        if not path:
            print("[ABORT] Planner found no path.")
            if not args.no_viz and args.save_html:
                save_path_html(arm, [q_start, q_goal], args.save_html)
            return
        print(f"Path: {len(path)} waypoints.")

        if not args.no_viz:
            visualize(arm, path,
                      save_png=args.save_viz or None,
                      save_html=args.save_html or None,
                      live=args.live_viz,
                      target=None if dual else target)

        if not confirm(f"\nExecute {len(path)}-waypoint path on REAL HARDWARE?"):
            print("Aborted before execution.")
            return

        if args.gesture_when=="before":
            _apply_gestures(robot,args, "before")

        print("Executing...")
        ctrl.execute_trajectory(path, side=args.side, dwell=0.0)
        print("Path executed.")

        if args.gesture_when == "after":
            _apply_gestures(robot, args, "after")
        # ---- post-exec diagnostics ----
        time.sleep(1.0)
        if dual:
            q_after = ctrl.get_both_arms_q()
            for name, slc, tgt, tgt_ori in [
                    ("left",  slice(0, 7),  target_l, TARGET_LEFT_ORI),
                    ("right", slice(7, 14), target_r, TARGET_RIGHT_ORI)]:
                sub = arm.left if name == "left" else arm.right
                ee_after = sub.get_end_effector_position(q_after[slc])
                ee_goal  = sub.get_end_effector_position(q_goal[slc])
                tot = np.linalg.norm(ee_after - tgt) * 1000
                drift = np.linalg.norm(ee_after - ee_goal) * 1000
                ik_e = np.linalg.norm(ee_goal - tgt) * 1000
                rms = np.sqrt(np.mean(np.degrees(q_after[slc] - q_goal[slc]) ** 2))
                print(f"\n[{name}]  EE after: {np.round(ee_after, 3)}")
                print(f"  IK plan err      : {ik_e:5.1f} mm")
                if tgt_ori is not None:
                    T_goal  = sub.get_end_effector_pose(q_goal[slc])
                    T_after = sub.get_end_effector_pose(q_after[slc])
                    ori_ik = np.degrees(np.linalg.norm(
                        _orientation_error(T_goal[:3, :3], tgt_ori)))
                    ori_after = np.degrees(np.linalg.norm(
                        _orientation_error(T_after[:3, :3], tgt_ori)))
                    print(f"  IK plan ori err  : {ori_ik:5.1f} deg")
                    print(f"  Total ori err    : {ori_after:5.1f} deg")
                print(f"  Joint tracking RMS: {rms:5.2f} deg")
                print(f"  Tracking EE drift : {drift:5.1f} mm")
                print(f"  Total EE err     : {tot:5.1f} mm")
        else:
            q_after = ctrl.get_current_q(args.side)
            ee_after = arm.get_end_effector_position(q_after)
            ee_goal = arm.get_end_effector_position(q_goal)
            ik_e = np.linalg.norm(ee_goal - target) * 1000
            drift = np.linalg.norm(ee_after - ee_goal) * 1000
            tot = np.linalg.norm(ee_after - target) * 1000
            rms = np.sqrt(np.mean(np.degrees(q_after - q_goal) ** 2))
            print(f"\nPost-exec EE         : {np.round(ee_after, 3)}")
            print(f"  IK plan error      : {ik_e:5.1f} mm")
            print(f"  Joint tracking RMS : {rms:5.2f} deg")
            print(f"  Tracking EE drift  : {drift:5.1f} mm")
            print(f"  Total EE error     : {tot:5.1f} mm")

        input("\nPress Enter to home... ")
        ctrl.home()
        time.sleep(1.0)

def _validate(arm, q_goal, target_a, target_b=None,
              target_a_ori=None, target_b_ori=None,
              ik_max_m=0.10, ik_max_ori_deg=20.0):
    """Single-arm: target_b=None. Dual-arm: target_a=left, target_b=right.
    Orientation targets are optional 3x3 rotations. When given, the
    orientation error is reported and bounded by ik_max_ori_deg.
    """
    if not arm.is_within_joint_limits(q_goal):
        return False, "goal outside joint limits"

    ik_max_ori_rad = np.radians(ik_max_ori_deg)
    parts = []

    if target_b is None:
        # ---- single arm ----
        fk = arm.get_end_effector_position(q_goal)
        err_p = float(np.linalg.norm(np.array(target_a) - fk))
        parts.append(f"pos {err_p*1000:.1f} mm")
        if target_a_ori is not None:
            T = arm.get_end_effector_pose(q_goal)
            err_o = float(np.linalg.norm(
                _orientation_error(T[:3, :3], target_a_ori)))
            parts.append(f"ori {np.degrees(err_o):.1f} deg")
            if err_o > ik_max_ori_rad:
                return False, (f"IK ori err {np.degrees(err_o):.1f} deg "
                               f"> {ik_max_ori_deg:.0f} deg")
        if err_p > ik_max_m:
            return False, (f"IK pos err {err_p*1000:.0f} mm "
                           f"> {ik_max_m*1000:.0f} mm")
    else:
        # ---- dual arm ----
        fk_l = arm.left.get_end_effector_position(q_goal[:7])
        fk_r = arm.right.get_end_effector_position(q_goal[7:])
        err_pl = float(np.linalg.norm(np.array(target_a) - fk_l))
        err_pr = float(np.linalg.norm(np.array(target_b) - fk_r))
        parts.append(f"L pos {err_pl*1000:.1f} mm")
        parts.append(f"R pos {err_pr*1000:.1f} mm")

        if target_a_ori is not None:
            T_l = arm.left.get_end_effector_pose(q_goal[:7])
            err_ol = float(np.linalg.norm(
                _orientation_error(T_l[:3, :3], target_a_ori)))
            parts.append(f"L ori {np.degrees(err_ol):.1f} deg")
            if err_ol > ik_max_ori_rad:
                return False, (f"IK L ori err {np.degrees(err_ol):.1f} deg "
                               f"> {ik_max_ori_deg:.0f} deg")
        if target_b_ori is not None:
            T_r = arm.right.get_end_effector_pose(q_goal[7:])
            err_or = float(np.linalg.norm(
                _orientation_error(T_r[:3, :3], target_b_ori)))
            parts.append(f"R ori {np.degrees(err_or):.1f} deg")
            if err_or > ik_max_ori_rad:
                return False, (f"IK R ori err {np.degrees(err_or):.1f} deg "
                               f"> {ik_max_ori_deg:.0f} deg")

        if max(err_pl, err_pr) > ik_max_m:
            return False, (f"IK pos err L={err_pl*1000:.0f} "
                           f"R={err_pr*1000:.0f} mm "
                           f"> {ik_max_m*1000:.0f} mm")

    if arm.check_collision(q_goal):
        return False, "goal in collision with obstacles"
    return True, "IK ok (" + ", ".join(parts) + ")"


if __name__ == "__main__":
    main()