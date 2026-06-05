#!/usr/bin/env python3
"""
tune_thumb_0.py
===============
Interactive REPL for moving thumb_0 (motor 0) only, with a live readout
of its measured angle. Other six motors stay at their startup position.

Use this to confirm motor 0 exists, what direction it moves, and how its
URDF limits map to visible thumb motion.

Commands at the prompt
----------------------
  <num>       absolute target in DEGREES   (e.g. "20", "-30", "0")
  <num>r      absolute target in RADIANS   (e.g. "0.5r", "-0.3r")
  +<num>      relative nudge in degrees    (e.g. "+10")
  -<num>      relative nudge in degrees    (e.g. "-15")
  s           status snapshot
  o           open position (q=0)
  v           toggle live readout on/off
  h           help
  q           quit (releases motors)

Run
---
  python3 tune_thumb_0.py left
  python3 tune_thumb_0.py right
"""

import math
import sys
import time
import threading
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from g1_hand_controller import (
    G1HandController, JOINT_LIMITS, JOINT_NAMES, MOTOR_MAX
)


LIVE_PRINT_HZ = 2.0          # how often the live readout fires


def tune_thumb_0(side: str = "left"):
    """
    Open a REPL that controls thumb_0 on the given hand while continuously
    printing its measured angle from HandState_.

    The other six finger motors are held at whatever position the hand was
    in when start() seeded its state from hardware.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    ChannelFactoryInitialize(0)
    ctrl = G1HandController(side)
    ctrl.start()
    time.sleep(0.5)   # give the seed loop a beat to settle

    # Local copy of the 7-DOF target. We only ever modify index 0; motors
    # 1-6 stay at their seeded values, so they never move.
    local_q = ctrl.get_joint_rad()
    lo, hi  = JOINT_LIMITS[side][0]      # thumb_0 limits
    t0      = time.monotonic()

    # ------------------------------------------------------------------
    # Background live readout
    # ------------------------------------------------------------------
    live_on = [True]   # mutable so the REPL can toggle it
    print_running = [True]

    def live_print():
        period = 1.0 / LIVE_PRINT_HZ
        while print_running[0]:
            if live_on[0]:
                state = ctrl.get_state()
                if state is not None and len(state.motor_state) >= 1:
                    meas    = state.motor_state[0].q
                    tgt     = local_q[0]
                    elapsed = time.monotonic() - t0
                    print(f"  [{elapsed:6.1f}s] thumb_0  "
                          f"meas={meas:+0.3f} rad ({math.degrees(meas):+6.1f}°)  "
                          f"target={tgt:+0.3f} rad ({math.degrees(tgt):+6.1f}°)")
            time.sleep(period)

    reader = threading.Thread(target=live_print, daemon=True, name="thumb_live")
    reader.start()

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    def banner():
        print(f"\n=== thumb_0 tuner ({side}) — joint '{JOINT_NAMES[0]}' ===")
        print(f"limits : {lo:+.3f} to {hi:+.3f} rad   "
              f"({math.degrees(lo):+.1f}° to {math.degrees(hi):+.1f}°)")
        print(f"seeded : {local_q[0]:+.3f} rad ({math.degrees(local_q[0]):+.1f}°)")
        print("commands: <deg> | <num>r | +<deg> | -<deg> | s | o | v | h | q\n")
    banner()

    # ------------------------------------------------------------------
    # REPL
    # ------------------------------------------------------------------
    def parse_target(cmd: str) -> float | None:
        """Return the new absolute target angle in rad, or None on parse error."""
        try:
            if cmd == "o":
                return 0.0
            if cmd.endswith("r"):
                return float(cmd[:-1])
            if cmd.startswith("+") or cmd.startswith("-"):
                # Relative degrees from CURRENT TARGET (not measured).
                # Keeps cumulative nudges predictable even mid-slew.
                return local_q[0] + math.radians(float(cmd))
            return math.radians(float(cmd))
        except ValueError:
            return None

    try:
        while True:
            try:
                raw = input("thumb> ")
            except EOFError:
                break
            cmd = raw.strip().lower()
            if not cmd:
                continue

            if cmd == "q":
                break

            if cmd == "h":
                banner()
                continue

            if cmd == "v":
                live_on[0] = not live_on[0]
                print(f"  live readout: {'on' if live_on[0] else 'off'}")
                continue

            if cmd == "s":
                state = ctrl.get_state()
                if state is None:
                    print("  no state yet")
                else:
                    meas = state.motor_state[0].q
                    print(f"  measured: {meas:+.3f} rad ({math.degrees(meas):+.1f}°)")
                    print(f"  target  : {local_q[0]:+.3f} rad "
                          f"({math.degrees(local_q[0]):+.1f}°)")
                continue

            target = parse_target(cmd)
            if target is None:
                print("  bad input (try e.g. '20', '-15', '0.5r', '+10', or 'h')")
                continue

            clipped = float(np.clip(target, lo, hi))
            if abs(clipped - target) > 1e-6:
                print(f"  clipped to limits: "
                      f"{target:+.3f} → {clipped:+.3f} rad")
            local_q[0] = clipped
            ctrl.set_joint_rad(local_q)
            print(f"  → thumb_0 target = {clipped:+.3f} rad "
                  f"({math.degrees(clipped):+.1f}°)")

    except KeyboardInterrupt:
        print("\n  interrupted")

    finally:
        print_running[0] = False
        reader.join(timeout=1.0)
        ctrl.stop()


if __name__ == "__main__":
    side = sys.argv[1] if len(sys.argv) > 1 else "left"
    tune_thumb_0(side)