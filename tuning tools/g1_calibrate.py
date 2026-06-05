#!/usr/bin/env python3
"""
g1_calibrate.py

Interactive tool to position the arm against a real-world reference
(e.g., the top-center of a trashcan) and read out the end-effector
position in torso frame. Use this to calibrate obstacles before
plugging them into arm.obstacles for path planning.

Workflow:
  1. Run, robot in 'Get ready' mode, suspended, hands free of obstacles
  2. Type joint commands until the rubber hand is touching the reference
     point on the obstacle
  3. Type 'ee' to read the current EE position; that is your reference
     point in torso frame
  4. Type 'obs W H D' (full sizes in m) to print the obstacle definition
     line you can paste into arm.obstacles, treating the current EE as
     the TOP-CENTER of the box

Commands:
  set NAME deg    e.g. set LSP -30      (absolute offset from init, deg)
  rel NAME ddeg   e.g. rel LE 5         (relative change, deg)
  ee              print current EE position
  q               print current joint angles in deg
  obs W H D       print obstacle line; W=Y, H=Z, D=X full sizes (m)
                  (matches your earlier 50x77x28 cm trashcan: obs 0.50 0.77 0.28)
  home            return arm to init pose
  quit
"""

import math
import time
import numpy as np

from myArm_v2 import G1Arm
from g1_arm_controller import G1ArmController
from g1_arm_live_tuner import JOINT_NAME, NAME_TO_JOINT


JOINT_LABELS = ["LSP", "LSR", "LSY", "LE ", "LWR", "LWP", "LWY"]


def print_state(ctrl, arm, side="left"):
    q = ctrl.get_current_q(side)
    ee = arm.get_end_effector_position(q)
    print(f"  q (deg): " + "  ".join(
        f"{l}={d:+6.1f}" for l, d in zip(JOINT_LABELS, np.degrees(q))))
    print(f"  EE (torso frame, m): "
          f"X={ee[0]:+.4f}  Y={ee[1]:+.4f}  Z={ee[2]:+.4f}")
    return q, ee


def print_obstacle_line(ee_xyz, w_y, h_z, d_x):
    """Treat ee_xyz as the TOP-CENTER of the obstacle box.
    w_y=size in Y (left-right), h_z=size in Z (height), d_x=size in X (depth)."""
    cx, cy, cz = float(ee_xyz[0]), float(ee_xyz[1]), float(ee_xyz[2] - h_z / 2.0)
    print()
    print("Paste into your code (e.g. before plan_and_execute_path):")
    print()
    print(f"arm.obstacles = np.array([")
    print(f"    [{cx:.4f}, {cy:.4f}, {cz:.4f}, 0, 0, 0, "
          f"{d_x:.3f}, {w_y:.3f}, {h_z:.3f}],")
    print(f"])")
    print()
    print(f"  Box center in torso frame: ({cx:+.3f}, {cy:+.3f}, {cz:+.3f})  m")
    print(f"  Box full sizes (X, Y, Z) : ({d_x:.3f}, {w_y:.3f}, {h_z:.3f})  m")
    print(f"  Box X range              : "
          f"{cx - d_x/2:+.3f}  to  {cx + d_x/2:+.3f}")
    print(f"  Box Y range              : "
          f"{cy - w_y/2:+.3f}  to  {cy + w_y/2:+.3f}")
    print(f"  Box Z range              : "
          f"{cz - h_z/2:+.3f}  to  {cz + h_z/2:+.3f}  "
          f"(top = {cz + h_z/2:+.3f})")


def main():
    side = "left"
    print("Robot suspended, in 'Get ready' main op mode? E-stop ready?")
    input("Press Enter to connect... ")

    # Slow + stiff: precise positioning, slow speed so you can stop midway if
    # the arm is heading somewhere bad. Use ctrl-c to abort a slew at any time.
    with G1ArmController(max_arm_speed_dps=10.0,
                         kp_arm=120.0, kd_arm=2.0) as ctrl:
        print("\nMoving to clearance pose (LSR=+20 deg)...")
        q = ctrl.get_current_q(side)
        q[1] = math.radians(20.0)
        ctrl.move_arm_to(q, side=side)
        time.sleep(0.5)

        arm = G1Arm()
        # Don't let any leftover obstacle interfere; calibration only reads state
        arm.obstacles = np.empty((0, 9))

        print("\nReady. Type commands to position the hand against your reference point.")
        print("Type 'help' for the command list.")
        print()
        print_state(ctrl, arm, side)

        last_ee = None
        try:
            while True:
                line = input("> ").strip()
                if not line:
                    continue
                parts = line.split()
                cmd = parts[0].lower()

                if cmd in ("quit", "exit"):
                    break

                elif cmd == "help":
                    print("  set NAME deg     (e.g. set LSP -30)")
                    print("  rel NAME ddeg    (e.g. rel LE 5)")
                    print("  ee               print current EE")
                    print("  q                print joint angles")
                    print("  obs W H D        print obstacle line (full sizes, m)")
                    print("                   W=Y, H=Z, D=X")
                    print("  home             return to init pose")
                    print("  quit")

                elif cmd == "ee":
                    _, last_ee = print_state(ctrl, arm, side)

                elif cmd == "q":
                    print_state(ctrl, arm, side)

                elif cmd == "home":
                    ctrl.home()
                    time.sleep(0.5)
                    print_state(ctrl, arm, side)

                elif cmd == "set" and len(parts) == 3:
                    try:
                        name = parts[1].upper()
                        deg = float(parts[2])
                    except ValueError:
                        print("  Bad number.")
                        continue
                    if name not in NAME_TO_JOINT:
                        print(f"  Unknown joint: {name}")
                        continue
                    ctrl.tuner.set_abs_offset_deg(name, deg)
                    time.sleep(0.3)
                    _, last_ee = print_state(ctrl, arm, side)

                elif cmd == "rel" and len(parts) == 3:
                    try:
                        name = parts[1].upper()
                        ddeg = float(parts[2])
                    except ValueError:
                        print("  Bad number.")
                        continue
                    if name not in NAME_TO_JOINT:
                        print(f"  Unknown joint: {name}")
                        continue
                    ctrl.tuner.add_rel_offset_deg(name, ddeg)
                    time.sleep(0.3)
                    _, last_ee = print_state(ctrl, arm, side)

                elif cmd == "obs" and len(parts) == 4:
                    if last_ee is None:
                        _, last_ee = print_state(ctrl, arm, side)
                    try:
                        w_y = float(parts[1])
                        h_z = float(parts[2])
                        d_x = float(parts[3])
                    except ValueError:
                        print("  Bad number.")
                        continue
                    print_obstacle_line(last_ee, w_y, h_z, d_x)

                else:
                    print("  Unknown command. Type 'help'.")

        except KeyboardInterrupt:
            print("\nInterrupted.")

        print("\nReturning to init pose...")
        ctrl.home()
        time.sleep(1.0)


if __name__ == "__main__":
    main()


