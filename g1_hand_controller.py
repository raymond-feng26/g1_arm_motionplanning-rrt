#!/usr/bin/env python3
"""
g1_hand_controller.py
=====================
Controls the Unitree G1 dexterous hands (dex3) over DDS.

The dex3 hand is an electrically separate subsystem from the body:
it has its own MCU, its own DDS topic, and its own 1-second watchdog.
L2+B damping on the remote only affects the body (rt/arm_sdk); the hand
keeps holding pose until you actively command zero torque OR stop
publishing for >1 s.

Topics
------
  Publish  : rt/dex3/left/cmd   or  rt/dex3/right/cmd   (HandCmd_)
  Subscribe: rt/dex3/left/state or  rt/dex3/right/state (HandState_)

Motor ordering (Unitree SDK, 0-indexed) — verify with sweep_test() before use:
  0: thumb_0   (abduction — swings thumb across the palm; subtle visual motion)
  1: thumb_1   (MCP flex)
  2: thumb_2   (IP flex, tip)
  3: index_0   (MCP)
  4: index_1   (PIP)
  5: middle_0  (MCP)
  6: middle_1  (PIP)

Usage
-----
    ChannelFactoryInitialize(0)
    ctrl = G1HandController("left")
    ctrl.start()

    ctrl.set_gesture("open")
    time.sleep(1.5)
    ctrl.set_gesture("close")
    time.sleep(1.5)

    ctrl.stop()   # stops control loop, then releases motors cleanly
"""

import math
import sys
import time
import threading
import numpy as np

from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, MotorCmd_

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOTOR_MAX  = 7
CONTROL_HZ = 100.0
DT         = 1.0 / CONTROL_HZ

# Default PD gains (mNm/rad, mNm/(100 rad/s)) — from Unitree C++ example
DEFAULT_KP = 2.5
DEFAULT_KD = 0.3

# Conservative slew cap to keep fingers from snapping
MAX_SLEW_RAD_S = math.radians(150.0)   # ~100 deg/s

# Watchdog disengage time on the hand MCU; we hold zero-gain for this long
# plus margin so the firmware definitively drops authority.
WATCHDOG_S       = 1.0
RELEASE_HOLD_S   = 1.5


def _make_mode(motor_id: int) -> int:
    """Pack RIS_Mode_t: id(4 bits) | status(3 bits)<<4 | timeout(1 bit)<<7.

    status=1 = FOC mode (motor active, obeys kp/kd/tau)
    timeout=1 = enable 1-second command watchdog
    """
    return (motor_id & 0x0F) | (0x01 << 4) | (0x01 << 7)


# ---------------------------------------------------------------------------
# Joint limits from URDF  [lower_rad, upper_rad]
# Motor order: [thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1]
# ---------------------------------------------------------------------------

JOINT_LIMITS = {
    "left": [
        (-1.04719755,  1.04719755),   # thumb_0  (Y-axis abduction)
        (-0.72431163,  1.04719755),   # thumb_1
        ( 0.0,         1.74532925),   # thumb_2
        (-1.57079632,  0.0),          # index_0
        (-1.74532925,  0.0),          # index_1
        (-1.57079632,  0.0),          # middle_0
        (-1.74532925,  0.0),          # middle_1
    ],
    "right": [
        # Right hand is the mirror: thumb_0 keeps the same Y-axis direction,
        # everything else has flipped signs.
        (-1.04719755,  1.04719755),   # thumb_0  (same as left)
        (-1.04719755,  0.72431163),   # thumb_1
        (-1.74532925,  0.0),          # thumb_2
        ( 0.0,         1.57079632),   # index_0
        ( 0.0,         1.74532925),   # index_1
        ( 0.0,         1.57079632),   # middle_0
        ( 0.0,         1.74532925),   # middle_1
    ],
}

JOINT_NAMES = ["thumb_0", "thumb_1", "thumb_2",
               "index_0", "index_1", "middle_0", "middle_1"]


# ---------------------------------------------------------------------------
# Gesture library  (radians, LEFT hand)
#
# For both hands, q=0 on all joints = open / straight.
#   Left  finger joints have negative limits  → close = negative
#   Right finger joints have positive limits  → close = positive
# Right hand = left * NEGATE_MASK (flips sign on everything except thumb_0).
# ---------------------------------------------------------------------------

GESTURES_RAD = {
    "left":{
        "open":  np.array([ -0.046,  -0.72,  0.0,  0.0,   0.0,   0.0,   0.0]),
        "close": np.array([ -0.046,  1,  1.5, -1.57,  -1.5,  -1.57,  -1.5]),
        "pinch": np.array([ -0.046,  0.7,  1.2, -1.2,  -1.4,   0.0,   0.0]),
        "point": np.array([ -0.046,  0.8,  1.5,  0.0,   0.0,  -1.3,  -1.5]),
        "peace": np.array([ -0.046,  0.8,  1.5,  0.0,   0.0,   0.0,   0.0]),
        "grab": np.array([ -0.046,  0.0,  1.5,  0.0,   0.0,   -0.35,   -0.85]),
    },
    "right":{
        "open":  np.array([ -0.046,  0.72,  0.0,  0.0,   0.0,   0.0,   0.0]),
        "close": np.array([ -0.046, -1, -1.5,  1.57,   1.5,   1.57,   1.5]),
        "pinch": np.array([ -0.046, -0.7, -1.2,  1.2,   1.4,   0.0,   0.0]),
        "point": np.array([ -0.046, -0.8, -1.5,  0.0,   0.0,   1.3,   1.5]),
        "peace": np.array([ -0.046, -0.8, -1.5,  0.0,   0.0,   0.0,   0.0]),
        "grab": np.array([ -0.046,  0.0,  -1.5,  0.0,   0.0,   0.35,   0.85]),
    }
}


def gesture_to_rad(side: str, gesture_name: str) -> np.ndarray:
    """Return a 7-element joint-angle array (rad) for a named gesture."""
    table=GESTURES_RAD.get(side)
    if table is None:
        raise ValueError(f"side must be 'left' or 'right', got '{side}'")
    if gesture_name not in table:
        raise ValueError(f"Unknown gesture '{gesture_name}'. "
                         f"Choose from: {list(table)}")
    q = table[gesture_name].copy()
    limits = JOINT_LIMITS[side]
    return np.array([float(np.clip(q[i], lo, hi))
                     for i, (lo, hi) in enumerate(limits)])


# ---------------------------------------------------------------------------
# MotorCmd_ builders
# ---------------------------------------------------------------------------

def _build_motor_cmd(motor_id: int, q_rad: float,
                     kp: float = DEFAULT_KP,
                     kd: float = DEFAULT_KD,
                     tau: float = 0.0) -> MotorCmd_:
    """Active PD command — motor will track q_rad with the given gains."""
    return MotorCmd_(
        mode    = _make_mode(motor_id),
        q       = q_rad,
        dq      = 0.0,
        tau     = tau,
        kp      = kp,
        kd      = kd,
        reserve = 0,
    )


def _build_release_cmd(motor_id: int) -> MotorCmd_:
    """Zero-gain FOC command — motor stays powered but applies no torque.
    Combined with stopping the publisher, the 1-second watchdog disengages
    the motor fully."""
    return MotorCmd_(
        mode    = _make_mode(motor_id),
        q       = 0.0,
        dq      = 0.0,
        tau     = 0.0,
        kp      = 0.0,
        kd      = 0.0,
        reserve = 0,
    )


# ---------------------------------------------------------------------------
# G1HandController
# ---------------------------------------------------------------------------

class G1HandController:
    """
    Runs a 100 Hz background thread that continuously publishes hand joint
    commands to rt/dex3/<side>/cmd.

    Public API
    ----------
    start()                       — connect, seed state, begin control loop
    stop()                        — stop loop, then release motors (clean)
    release()                     — make hand go limp (only safe when loop is stopped)
    set_joint_rad(q)              — move to raw 7-DOF joint config
    set_gesture(name)             — move to a named gesture preset
    get_joint_rad()               — current commanded joint angles
    get_state()                   — latest HandState_ from hardware
    sweep_test(motor_id, deg)     — single-motor validation sweep
    """

    def __init__(self, side: str,
                 kp: float = DEFAULT_KP,
                 kd: float = DEFAULT_KD):
        """
        Parameters
        ----------
        side : "left" or "right"
        kp   : stiffness gain (mNm/rad)
        kd   : damping gain   (mNm/(100 rad/s))
        """
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self.side   = side
        self.kp     = kp
        self.kd     = kd
        self.limits = JOINT_LIMITS[side]

        self._target_q  = None   # seeded from hardware on start()
        self._current_q = None

        self._state_lock   = threading.Lock()
        self._latest_state: HandState_ | None = None

        self._running    = False
        self._thread     = None
        self._publisher  = None
        self._subscriber = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, timeout: float = 2.0):
        """
        Connect to DDS, seed joint state from hardware so the hand doesn't
        jerk on startup, then start the control loop.

        If no HandState_ arrives within `timeout`, falls back to zeros and
        prints a warning — this will move the hand toward the open pose.
        """
        cmd_topic   = f"rt/dex3/{self.side}/cmd"
        state_topic = f"rt/dex3/{self.side}/state"

        self._publisher  = ChannelPublisher(cmd_topic,  HandCmd_)
        self._subscriber = ChannelSubscriber(state_topic, HandState_)
        self._publisher.Init()
        self._subscriber.Init(self._state_callback, 10)

        print(f"[G1HandController:{self.side}] waiting for HandState_...",
              end=" ", flush=True)
        deadline = time.monotonic() + timeout
        seeded = False
        while time.monotonic() < deadline:
            with self._state_lock:
                state = self._latest_state
            if state is not None and len(state.motor_state) >= MOTOR_MAX:
                q0 = np.array([state.motor_state[i].q for i in range(MOTOR_MAX)],
                              dtype=float)
                self._current_q = q0.copy()
                self._target_q  = q0.copy()
                print(f"seeded from hardware: {np.round(q0, 3)}")
                seeded = True
                break
            time.sleep(0.05)

        if not seeded:
            print("timeout — falling back to zeros (hand may move!)")
            self._current_q = np.zeros(MOTOR_MAX)
            self._target_q  = np.zeros(MOTOR_MAX)

        self._running = True
        self._thread = threading.Thread(
            target=self._control_loop, daemon=True,
            name=f"hand_{self.side}"
        )
        self._thread.start()
        print(f"[G1HandController:{self.side}] control loop started.")

    def stop(self):
        """
        Graceful shutdown. Order matters:
          1. Stop the 100 Hz control loop and wait for it to exit.
          2. THEN publish zero-gain commands to release motors.

        Doing release-before-stop creates a race: the still-running control
        loop overwrites the release packets with stiff PD commands at 100 Hz
        and the hand never actually goes limp.
        """
        if not self._running:
            return
        print(f"[G1HandController:{self.side}] stopping control loop...")
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.release()
        print(f"[G1HandController:{self.side}] stopped.")

    def release(self):
        """
        Make the hand go limp.

        Publishes zero-gain FOC commands for RELEASE_HOLD_S (longer than the
        firmware's 1-second watchdog), which:
          (a) keeps applied torque at zero while still in command,
          (b) lets the watchdog fully disengage motors at the end.

        NOT safe to call while the control loop is running — it will race.
        stop() calls this for you in the correct order.
        """
        if self._publisher is None:
            return
        if self._running:
            print(f"[G1HandController:{self.side}] WARNING: release() called "
                  f"while control loop is running. Call stop() instead.")

        motor_cmds = [_build_release_cmd(i) for i in range(MOTOR_MAX)]
        msg = HandCmd_(motor_cmd=motor_cmds, reserve=[0, 0, 0, 0])

        # Hold zero-gain at 100 Hz for >watchdog so motors definitely disengage.
        end = time.monotonic() + RELEASE_HOLD_S
        while time.monotonic() < end:
            self._publisher.Write(msg)
            time.sleep(DT)
        print(f"[G1HandController:{self.side}] motors released "
              f"({RELEASE_HOLD_S:.1f}s zero-gain + watchdog disengage).")

    # ------------------------------------------------------------------
    # Public control API
    # ------------------------------------------------------------------

    def set_joint_rad(self, q: np.ndarray):
        """
        Command all 7 joints to the given angles (rad). The controller slews
        toward this target at MAX_SLEW_RAD_S. Values are clipped to URDF
        joint limits automatically.
        """
        q = np.array(q, dtype=float)
        if q.shape != (MOTOR_MAX,):
            raise ValueError(f"q must be length {MOTOR_MAX}, got {q.shape}")
        clipped = np.array([np.clip(q[i], lo, hi)
                            for i, (lo, hi) in enumerate(self.limits)])
        if not np.allclose(q, clipped, atol=1e-4):
            print(f"[G1HandController:{self.side}] WARNING: "
                  f"clipped joint target to limits")
        self._target_q = clipped

    def set_gesture(self, name: str):
        """Move to a named gesture. See LEFT_GESTURES_RAD for available names."""
        q = gesture_to_rad(self.side, name)
        self.set_joint_rad(q)
        print(f"[G1HandController:{self.side}] → gesture '{name}'")

    def get_joint_rad(self) -> np.ndarray:
        """Current slewed (commanded) joint angles."""
        return self._current_q.copy()

    def get_state(self) -> HandState_ | None:
        """Latest HandState_ message (None if not yet received)."""
        with self._state_lock:
            return self._latest_state

    # ------------------------------------------------------------------
    # Sweep test
    # ------------------------------------------------------------------

    def sweep_test(self, motor_id: int = 0,
                   deg: float = 20.0,
                   duration: float = 2.0):
        """
        Move a single motor ±deg degrees from the open pose to verify the
        correct finger segment moves in the expected direction.

        Tip for motor 0 (thumb_0): use a bigger sweep (~45°) and start from
        the open gesture — thumb abduction is subtle from a closed pose.
        """
        amplitude = math.radians(deg)
        lo, hi    = self.limits[motor_id]
        print(f"\n[sweep_test:{self.side}] motor {motor_id} "
              f"({JOINT_NAMES[motor_id]})  ±{deg}°")

        base = gesture_to_rad(self.side, "open").copy()
        self.set_joint_rad(base)
        time.sleep(duration)

        # + direction
        target = base.copy()
        target[motor_id] = float(np.clip(base[motor_id] + amplitude, lo, hi))
        print(f"  → +{deg}°  ({target[motor_id]:+.3f} rad)")
        self.set_joint_rad(target)
        time.sleep(duration)

        # - direction
        target[motor_id] = float(np.clip(base[motor_id] - amplitude, lo, hi))
        print(f"  → -{deg}°  ({target[motor_id]:+.3f} rad)")
        self.set_joint_rad(target)
        time.sleep(duration)

        # Return to neutral
        self.set_joint_rad(base)
        time.sleep(duration)
        print(f"[sweep_test:{self.side}] motor {motor_id} done.\n")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _state_callback(self, msg: HandState_):
        with self._state_lock:
            self._latest_state = msg

    def _control_loop(self):
        """100 Hz: slew _current_q → _target_q, publish HandCmd_."""
        max_step = MAX_SLEW_RAD_S * DT

        while self._running:
            if self._current_q is None or self._target_q is None:
                time.sleep(DT)
                continue

            t0 = time.monotonic()

            # Slew
            delta = self._target_q - self._current_q
            step  = np.clip(delta, -max_step, max_step)
            self._current_q = self._current_q + step

            # Publish
            motor_cmds = [
                _build_motor_cmd(i, self._current_q[i], self.kp, self.kd)
                for i in range(MOTOR_MAX)
            ]
            msg = HandCmd_(motor_cmd=motor_cmds, reserve=[0, 0, 0, 0])
            self._publisher.Write(msg)

            # Pace
            sleep_t = DT - (time.monotonic() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)


# ---------------------------------------------------------------------------
# Standalone demos
# ---------------------------------------------------------------------------

def run_gesture_demo(side: str = "left"):
    """Cycle through all gestures on the specified hand."""
    ChannelFactoryInitialize(0)
    ctrl = G1HandController(side)
    ctrl.start()

    sequence = ["open", "close"]  # 
    try:
        for gesture in sequence:
            print(f"\n--- {gesture.upper()} ---")
            ctrl.set_gesture(gesture)
            time.sleep(2.0)
    finally:
        ctrl.stop()


def run_sweep_demo(side: str = "left"):
    """Interactive per-motor sweep — verifies motor index → finger mapping."""
    ChannelFactoryInitialize(0)
    ctrl = G1HandController(side)
    ctrl.start()
    time.sleep(0.5)
    try:
        for mid in range(MOTOR_MAX):
            # thumb_0 (abduction) is subtle — use a bigger sweep
            deg = 45.0 if mid == 0 else 20.0
            ctrl.sweep_test(motor_id=mid, deg=deg, duration=2.0)
            ans = input("  Correct finger/direction? [y/n]: ").strip().lower()
            if ans != "y":
                print(f"  !! Motor {mid} ({JOINT_NAMES[mid]}) needs correction")
    finally:
        ctrl.stop()


if __name__ == "__main__":
    side = sys.argv[1] if len(sys.argv) > 1 else "left"
    if "--sweep" in sys.argv:
        run_sweep_demo(side)
    else:
        run_gesture_demo(side)