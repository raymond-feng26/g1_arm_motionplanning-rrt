#!/usr/bin/env python3
"""
pull_left_run.py
================
Vision-guided PULL-LEFT, executed through the choreography engine, behind a
single confirm gate. Separated out from grab_run.py so the grab and the pull
are independent, readable entry points.

Use this for a box jammed on the robot's RIGHT (e.g. against a wall), too far
out to grasp directly: the RIGHT hand closes, plans to a staging pose
behind/below/left of the box tag, reaches in, and (intended to) drag the box
toward +y (the robot's left). It then STOPS — no grab, hand left in place;
pulling the hand back out / re-grasping is a follow-up.

Same front-end as grab_run.py
-----------------------------
    tag -> box -> obstacle worlds          (grasp_scene.snapshot_scene)
        |
        v
    build_pull_left(...)   right close -> staging (RRT) -> reach in -> [drag]
        |
        v
    run_sequence(...)      one confirm, then runs

NOTE — the +y drag itself is currently COMMENTED OUT in build_pull_left
(choreographies.py): this runs the approach + reach-in and stops there. That is
intentional (the hand-extraction strategy after the drag isn't settled yet).
Re-enable the drag block in build_pull_left when you're ready.

The right wrist-yaw motor is LOCKED (broken). The approach targets orientation
identity, which IK can't fully meet with the wrist pre-positioned, so watch the
approach orientation on hardware (preflight only checks limits + collision).
This is exactly the kind of new planning-path behavior to flag to Professor
Kantaros before a hardware run.

Speed
-----
MAX_ARM_SPEED_DPS starts low. Ramp it once validated, then re-check the
tracking-abort threshold for the new speed.
"""

import numpy as np

from myArm_dual import G1DualArm
from g1_robot import G1Robot
from g1_plan_run import _default_obstacles
from g1_choreograph import run_sequence, preflight_validate
from choreographies import build_pull_left

from perception_client import PerceptionClient
from grasp_scene import (
    snapshot_scene, render_planned, PERCEPTION_HOST, PERCEPTION_PORT,
)
from gravity_comp import GravityComp


# --- tunables ---------------------------------------------------------------
MAX_ARM_SPEED_DPS  = 80.0      # conservative first-bringup slew; ramp after validation
GRAVITY_COMP_SCALE = 1.0
DRAG_M             = 0.30      # total +y drag distance (only used once the drag
                              # block in build_pull_left is re-enabled)


def confirm(msg):
    return input(f"\n>>> {msg} [y/N]: ").strip().lower() == "y"


def main():
    np.set_printoptions(precision=3, suppress=True)

    print("[pull] connecting to G1 (arm + both hands)...")
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
        robot.arm.locked_right = list(arm.right.locked_joints.keys())

        # 2. Snapshot the scene from the live tags (client kept alive) -------
        client = PerceptionClient(host=PERCEPTION_HOST, port=PERCEPTION_PORT)
        client.start()
        scene = snapshot_scene(client)
        if scene is None:
            return
        scene.print_summary()

        # 3. Build the pull-left sequence + preflight -----------------------
        pull_steps = build_pull_left(arm, scene.T_base_box,
                                     scene.obs_full, scene.obs_no_box,
                                     drag_m=DRAG_M)
        arm.obstacles = np.array(_default_obstacles(), dtype=float)
        ok, _ = preflight_validate(arm, pull_steps, obstacle_names=scene.names_for)
        if not ok:
            print("[ABORT] pull-left preflight failed.")
            return

        # 4. Single gated run ----------------------------------------------
        if not confirm("Run PULL-LEFT (right hand to box, then drag), then stop?"):
            print("[pull] declined — nothing moved.")
            return

        pull_record = []
        run_sequence(robot, arm, pull_steps, record=pull_record)
        render_planned(arm, pull_record, "pull")
        print("\n[pull] complete — stopping (hand left in place; extraction TBD).")

    # `with` exit -> hands release, arm PD fades (safe disconnect).
    print("[pull] disconnected safely.")


if __name__ == "__main__":
    main()