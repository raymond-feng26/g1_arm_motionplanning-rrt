#!/usr/bin/env python3
"""
grab_run.py
===========
Full vision-guided grab + lift + put-down, executed through the choreography
engine, behind confirm gates. This is THE main grasp deliverable.

Connector between the perception/grasp pipeline and the choreography layer:

    tag -> box -> 4 grasp targets + 2 obstacle worlds   (grasp_scene.snapshot_scene)
        |
        v
    build_grab(...)        clearance / approach / grasp+grab / lift steps
        |
        v
    run_sequence(...)      ONE confirm before the first phase, then auto-runs
        |
        v
    build_put_down(...)    (optional, gated) place / push right / retreat to rest

Staged execution
-----------------
The grab runs in ONE run_sequence call. A single [y/N] before the first phase
(clearance); once confirmed, all phases (clearance / approach / close-in / lift)
run automatically, paced by each step's `settle`. last_commanded threads across
the whole sequence, so each phase's IK seeds from the previous phase's
*commanded* endpoint rather than a re-read (gravity-sagged) measured pose.
Declining the single gate runs nothing and leaves the arm where it is.

Validation & plan HTML
----------------------
Before any motion, IK is solved and validated for the pre-grasp and grasp goals
(reachable, in-limits, collision-free in their respective worlds), and the
current pose is checked against the box world. There is no dry pre-plan:
preflight_validate re-checks every goal, and run_sequence plans each leg at
execution time behind its gate. The interactive plotly HTML is rendered AFTER
the run (render_planned), straight from the paths run_sequence already streamed
— so the only RRT cost is the real one.

After the lift
--------------
After a successful lift there is a SECOND gate: PUT-DOWN. Accept it to run the
place-and-clear choreography (one run_sequence): lower the box onto the table
and release; the RIGHT hand withdraws, re-closes, and plans back to rest; the
LEFT hand nudges the box to the robot's right (-y), backs out, re-closes, and
plans back to rest. All returns plan in the box-free world (obs_no_box), so no
box re-perception is needed. Decline it (or if it fails preflight) and the
script holds the grip until you press Enter — exiting then releases the hands
and drops the box. A completed put-down has already placed the box, so that
release prompt is skipped. Walking is out of scope.

Pull-left (dragging a box jammed on the right toward the robot's left) is a
separate behavior and lives in its own entry point: pull_left_run.py.

Speed
-----
MAX_ARM_SPEED_DPS starts low for first bring-up. Ramp it up once the full move
is validated, and re-check the tracking-abort threshold for the new speed
(execute_trajectory defaults to 30 deg).
"""

import numpy as np

from myArm_dual import G1DualArm
from g1_robot import G1Robot
from g1_plan_run import _default_obstacles
from g1_choreograph import run_sequence, preflight_validate
from choreographies import build_grab, build_put_down

from perception_client import PerceptionClient
from grasp_scene import (
    snapshot_scene, solve_and_validate, report_ee, render_planned,
    NAMES_FULL, NAMES_NO_BOX, PERCEPTION_HOST, PERCEPTION_PORT,
)
from gravity_comp import GravityComp


# --- tunables ---------------------------------------------------------------
MAX_ARM_SPEED_DPS  = 80.0      # conservative first-bringup slew; ramp after validation
LIFT_SP_DEG        = -30.0     # shoulder-pitch delta at lift (deg); negative raises
GRAVITY_COMP_SCALE = 1.0


def confirm(msg):
    return input(f"\n>>> {msg} [y/N]: ").strip().lower() == "y"


def main():
    np.set_printoptions(precision=3, suppress=True)

    print("[grab] connecting to G1 (arm + both hands)...")
    gc = GravityComp(urdf_path="g1_29dof_with_hand.urdf", scale=GRAVITY_COMP_SCALE)

    with G1Robot(arm=True, left_hand=True, right_hand=True,
                 max_arm_speed_dps=MAX_ARM_SPEED_DPS, gravity_comp=gc) as robot:
        ctrl = robot.arm

        # 1. Measured start + RWY lock --------------------------------------
        q_start  = ctrl.get_current_q(side="both")
        rwy_lock = float(q_start[13])
        print(f"  start q  L: {np.degrees(q_start[:7]).round(1)} deg")
        print(f"           R: {np.degrees(q_start[7:]).round(1)} deg")
        print(f"  right RWY (joint 13) locked at {np.degrees(rwy_lock):+.1f} deg")

        arm = G1DualArm(locked_joints_right={6: rwy_lock})
        # Mask the locked joint from the executor's tracking-error abort
        # (a locked motor can't track, by definition).
        robot.arm.locked_right = list(arm.right.locked_joints.keys())

        # 2. Snapshot the scene from the live tags --------------------------
        #    The perception client is created here and kept alive for the whole
        #    run (its read thread is a daemon). snapshot_scene only waits + builds
        #    the box/table poses, grasp targets, and the two obstacle worlds.
        client = PerceptionClient(host=PERCEPTION_HOST, port=PERCEPTION_PORT)
        client.start()
        scene = snapshot_scene(client)
        if scene is None:
            return
        scene.print_summary()

        # 3. Up-front IK validation (no motion, no RRT) --------------------
        #    Prove both legs are reachable and collision-free at the goal before
        #    committing. preflight_validate (step 4) re-checks every goal;
        #    run_sequence plans each leg at execution behind the confirm gate. No
        #    dry pre-plan — plan HTML is rendered AFTER the run from the paths
        #    run_sequence actually streamed (render_planned), so no duplicate cost.
        print("\n=== IK validation ===")
        q_pre = solve_and_validate(arm, scene.targets["left_pre"], scene.targets["right_pre"],
                                   q_start, scene.obs_full, "PREGRASP",
                                   obstacle_names=NAMES_FULL, r_l=scene.R_yaw, r_r=np.eye(3))
        if q_pre is None:
            return
        q_g   = solve_and_validate(arm, scene.targets["left_grasp"], scene.targets["right_grasp"],
                                   q_pre, scene.obs_no_box, "GRASP   ",
                                   obstacle_names=NAMES_NO_BOX, r_l=scene.R_yaw, r_r=np.eye(3))
        if q_g is None:
            return

        arm.obstacles = np.array(scene.obs_full, dtype=float)
        if arm.check_collision(q_start):
            print("[ABORT] current arm pose already collides with box/table.")
            return

        # 4. Build the choreography + preflight -----------------------------
        steps = build_grab(scene.targets, scene.obs_full, scene.obs_no_box,
                            clearance=True, lift_sp_deg=LIFT_SP_DEG,
                            R_left=scene.R_yaw, R_right=np.eye(3))

        # Baseline world for run_sequence (per-step `obstacles` override it; this
        # is just the keyless fallback / sub-arm sync seed).
        arm.obstacles = np.array(_default_obstacles(), dtype=float)
        ok, _ = preflight_validate(arm, steps, obstacle_names=scene.names_for)
        if not ok:
            print("[ABORT] preflight validation failed.")
            return

        # 5. Single gated run ----------------------------------------------
        #    One run_sequence over the whole grab. ONE [y/N] before the first
        #    phase (clearance); once confirmed, all phases run automatically.
        #    Inter-phase pacing is the per-step `settle` (tune in build_grab).
        #    EE is reported (print-only) for the box-critical phases.
        print(f"\n[grab] {len(steps)} steps. One confirm, then auto-runs.")
        state = {"prev_label": None, "gripped": False, "stopped": False}

        def _report_prev():
            pl = state["prev_label"]
            if pl is None:
                return
            if "approach" in pl:
                report_ee(arm, ctrl.get_current_q("both"),
                          scene.targets["left_pre"], scene.targets["right_pre"],
                          "AFTER APPROACH (pre-grasp)")
            elif pl.startswith("close in"):
                report_ee(arm, ctrl.get_current_q("both"),
                          scene.targets["left_grasp"], scene.targets["right_grasp"],
                          "AFTER GRASP")

        def gate(gi, n, group):
            _report_prev()  # report the group that just finished (print-only)
            label = group[0].get("label", f"group {gi+1}")
            # Confirm ONCE, before the first group. Later groups auto-run.
            if gi == 0:
                if not confirm(f"Start grab? First stage: {label} "
                               f"(then {n-1} more auto)"):
                    state["stopped"] = True
                    return False
            state["prev_label"] = label
            if group[-1].get("gestures_after", {}).get("left") == "grab":
                state["gripped"] = True
            return True

        grab_record = []
        run_sequence(robot, arm, steps, confirm=gate, record=grab_record)

        # Render the planned (RRT) legs of the grab AFTER the fact — the only
        # planned leg is the approach (clearance pose -> pre-grasp). No extra
        # planning: this draws the path run_sequence already streamed.
        render_planned(arm, grab_record, "grab")

        # Report EE for the final executed group (lift), if we didn't stop.
        if not state["stopped"]:
            _report_prev()
            print("\n[grab] lift complete — box lifted and held.")

            # 6. PUT-DOWN phase --------------------------------------------
            #    Place, release, retreat the right arm, nudge the box to the
            #    right with the left hand, back out, re-close and plan home — all
            #    six steps in one build_put_down / run_sequence. No box
            #    re-perception: the left return plans in obs_no_box. ONE confirm
            #    to start; once confirmed all stages auto-run. Declining holds the
            #    lift (box still gripped). Planned legs are rendered after the run.
            if state["gripped"] and confirm(
                    "Start PUT-DOWN (place box, release, push aside, retreat)?"):
                put_steps = build_put_down(arm, scene.obs_full, scene.obs_no_box)
                arm.obstacles = np.array(_default_obstacles(), dtype=float)
                ok, _ = preflight_validate(arm, put_steps, obstacle_names=scene.names_for)
                if not ok:
                    print("[ABORT] put-down preflight failed — holding the grip.")
                else:
                    put_record = []
                    run_sequence(robot, arm, put_steps, record=put_record)
                    render_planned(arm, put_record, "putdown")
                    print("\n[grab] put-down complete — box placed, arm at rest.")
                    state["gripped"] = False   # released onto the table

        # 7. Hold the grip until the user is ready to drop -----------------
        #    Reached holding the box only if the lift succeeded and the put-down
        #    was declined or failed; a completed put-down already placed the box,
        #    so this is skipped.
        if state["gripped"]:
            input("\nPress Enter to RELEASE and disconnect "
                  "(this drops the box)... ")

    # `with` exit -> hands release, arm PD fades (safe disconnect).
    print("[grab] disconnected safely.")


if __name__ == "__main__":
    main()