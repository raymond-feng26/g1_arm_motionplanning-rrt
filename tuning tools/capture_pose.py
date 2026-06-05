#!/usr/bin/env python3
"""
capture_pose.py
===============
Print the current arm joint angles in URDF-absolute units, formatted for
direct paste into _WAVE_NEUTRAL or any other choreography step's target_q.

Workflow
--------
  1. Pose the arm with the live tuner (or with the dex3 retracted manually,
     if the robot is suspended and you want to physically position it).
  2. While the pose is held, run:
         python3 capture_pose.py
     or for both arms:
         python3 capture_pose.py both

Notes
-----
- G1ArmController briefly takes arm_sdk authority on connect. If the live
  tuner is running in another terminal, it may fight us. For best results
  pause the live tuner (Ctrl-C is fine — the arm holds via damping) right
  before running this, or run this once on a quick poke and revert.
- The angles printed here are the ones to USE in choreographies.py.
"""

import sys
import numpy as np

from g1_arm_controller import G1ArmController


def capture(side: str):
    with G1ArmController() as ctrl:
        if side == "both":
            ql = ctrl.get_current_q("left")
            qr = ctrl.get_current_q("right")
            _print_pose("LEFT  arm", ql)
            print()
            _print_pose("RIGHT arm", qr)
        else:
            q = ctrl.get_current_q(side)
            _print_pose(f"{side.upper()} arm", q)



_JOINT_LABELS=["shoulder pitch", "shoulder roll", "shoulder yaw", "elbow", "wrist roll", "wrist pitch", "wrist yaw"]


def _print_pose(label: str, q: np.ndarray):
    deg = np.degrees(q)
    print(f"=== {label} (URDF-absolute) ===")
    for i, (val_deg, name) in enumerate(zip(deg, _JOINT_LABELS)):
        print(f"  joint {i} {name:<14s}: {val_deg:+8.2f} deg "
              f"({q[i]:+0.4f} rad)")
    print()
    print("Paste-ready (np.radians, degrees):")
    print("np.radians([")
    for val_deg, name in zip(deg, _JOINT_LABELS):
        print(f"    {val_deg:+7.2f},   # {name}")
    print("])")


if __name__ == "__main__":
    side = sys.argv[1] if len(sys.argv) > 1 else "left"
    if side not in ("left", "right", "both"):
        print("usage: python3 capture_pose.py [left|right|both]")
        sys.exit(1)
    capture(side)