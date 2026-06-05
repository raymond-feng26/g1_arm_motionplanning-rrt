#!/usr/bin/env python3
"""
g1_robot.py
===========
Skinny umbrella that bundles the arm controller and the two dex3 hand
controllers into a single lifecycle.

Why this exists
---------------
- Deterministic start order (arm first, which initializes DDS) and stop
  order (hands first → release grip before arm goes to damping).
- One try/finally in calling code instead of nested ones per sub-controller.
- emergency_release() for signal handlers.

What it deliberately is NOT
---------------------------
- No compound motion methods (move-and-grip etc.) — that's policy for plan_run.
- No trajectory awareness — the arm executor owns that.
- No planner/IK references — those live one level up in plan_run.

Usage
-----
    with G1Robot(arm=True, left_hand=True, right_hand=True) as robot:
        robot.left_hand.set_gesture("open")
        robot.arm.move_arm_to(q_target, side="left")
        robot.left_hand.set_gesture("close")

Construct with only what you need:
    with G1Robot(arm=True) as robot:                       # arm only
    with G1Robot(arm=True, left_hand=True) as robot:       # arm + left only
    with G1Robot(left_hand=True, right_hand=True) as robot: # hands only
"""

from __future__ import annotations

import sys
from typing import Optional

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from g1_arm_controller  import G1ArmController
from g1_hand_controller import G1HandController


class G1Robot:
    """Lifecycle umbrella for arm + (left|right) hands."""

    def __init__(self,
                 arm: bool = True,
                 left_hand: bool = False,
                 right_hand: bool = False,
                 channel_id: int = 0,
                 max_arm_speed_dps: float = 10.0,
                 kp_arm : float = 120.0,
                 kd_arm: float = 2.0,
                 gravity_comp=None):
        """
        Parameters
        ----------
        arm               : include the arm controller
        left_hand         : include the left dex3 controller
        right_hand        : include the right dex3 controller
        channel_id        : DDS channel id (default 0)
        max_arm_speed_dps : arm slew cap, passed to G1ArmController
        gravity_comp      : optional GravityComp instance; when given, the
                            arm controller's 100 Hz loop applies the
                            computed feedforward torque each tick. None
                            (default) means tau_ff=0 everywhere (legacy
                            behavior, no comp).
        """
        if not (arm or left_hand or right_hand):
            raise ValueError("G1Robot needs at least one sub-controller")

        self._channel_id = channel_id
        self._started    = False

        # Sub-controllers — None unless requested. Public so callers can
        # reach the underlying APIs directly: robot.arm.move_arm_to(...)
        self.arm: Optional[G1ArmController] = (
            G1ArmController(max_arm_speed_dps=max_arm_speed_dps,
                            channel_id=channel_id, kd_arm=kd_arm, kp_arm=kp_arm,
                            gravity_comp=gravity_comp)
            if arm else None
        )
        self.left_hand:  Optional[G1HandController] = (
            G1HandController("left")  if left_hand  else None
        )
        self.right_hand: Optional[G1HandController] = (
            G1HandController("right") if right_hand else None
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """
        Start sub-controllers in order. If any step fails, rolls back the
        ones that already started so we don't leave the robot in a partly-
        initialized state.

        Order rationale:
          1. Arm first — its connect() calls ChannelFactoryInitialize as a
             side effect, which the hands then ride on.
          2. Hands after — they assume DDS is already up.
          3. If no arm requested, we initialize DDS ourselves before hands.
        """
        if self._started:
            return

        started: list[str] = []
        try:
            if self.arm is not None:
                self.arm.connect()
                started.append("arm")
            else:
                ChannelFactoryInitialize(self._channel_id)

            if self.left_hand is not None:
                self.left_hand.start()
                started.append("left_hand")

            if self.right_hand is not None:
                self.right_hand.start()
                started.append("right_hand")

        except Exception:
            # Roll back in reverse safety order (hands first, arm last)
            self._stop_subset(reversed(started))
            raise

        self._started = True
        print("[G1Robot] all sub-controllers started.")

    def stop(self):
        """
        Stop in safety order: hands first (release grip), arm last (damping).
        Exceptions in any one sub-controller don't prevent the others from
        stopping — best-effort cleanup.
        """
        if not self._started:
            return
        self._stop_subset(["left_hand", "right_hand", "arm"])
        self._started = False
        print("[G1Robot] all sub-controllers stopped.")

    def emergency_release(self):
        """
        Best-effort, fast release for signal handlers or exception paths.
        Same order as stop() (hands → arm) but swallows everything and
        leaves _started in whatever state it was — assume the caller is
        on the way out.
        """
        self._stop_subset(["left_hand", "right_hand", "arm"])

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "G1Robot":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False   # don't suppress the exception

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _stop_subset(self, names):
        """Stop the named sub-controllers, swallowing per-controller errors."""
        for name in names:
            ctrl = getattr(self, name, None)
            if ctrl is None:
                continue
            try:
                # Arm uses connect/disconnect; hands use start/stop.
                if name == "arm":
                    ctrl.disconnect()
                else:
                    ctrl.stop()
            except Exception as exc:
                print(f"[G1Robot] error stopping {name}: {exc!r}",
                      file=sys.stderr)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Bring everything up, hold for 2s, bring it down — verifies lifecycle."""
    import time

    parts = sys.argv[1:] or ["arm", "left", "right"]
    with G1Robot(
        arm        = "arm"   in parts,
        left_hand  = "left"  in parts,
        right_hand = "right" in parts,
    ) as robot:
        print("[smoke] robot up — holding 3s")
        time.sleep(3.0)
    print("[smoke] done.")