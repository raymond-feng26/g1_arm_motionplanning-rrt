#!/usr/bin/env python3
"""
g1_arm_controller.py

Programmatic wrapper around g1_arm_live_tuner.LiveTuner.

Exposes the arm in *absolute joint angles (rad)*, in the same joint order
your myArm.G1Arm uses:
    [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
     wrist_roll, wrist_pitch, wrist_yaw]

This means you can take IK output straight from myArm and pass it here.

Usage (high level):

    from g1_arm_controller import G1ArmController

    with G1ArmController(max_arm_speed_dps=10.0) as ctrl:
        q_now = ctrl.get_current_q("left")
        ctrl.move_arm_to(some_q_target, side="left")

Safety: speed is rate-limited by MAX_ARM_DEG_PER_SEC inside the live tuner.
Pass max_arm_speed_dps=10.0 (or lower) for first hardware runs.
"""

import math
import time
import numpy as np

# Reuse everything from your existing tuner script
import g1_arm_live_tuner as glt
from g1_arm_live_tuner import (
    LiveTuner,
    G1JointIndex,
    JOINT_NAME,
    NAME_TO_JOINT,
    ARM_JOINTS,
    WAIST_JOINTS,
    CONTROLLED_JOINTS,
    d2r,
    r2d,
)
from unitree_sdk2py.core.channel import ChannelFactoryInitialize


# --- Joint index lists matching myArm.py joint ordering ---
# myArm.py order:  [pitch, roll, yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw]

LEFT_ARM_INDICES = [
    G1JointIndex.LeftShoulderPitch,   # 15
    G1JointIndex.LeftShoulderRoll,    # 16
    G1JointIndex.LeftShoulderYaw,     # 17
    G1JointIndex.LeftElbow,           # 18
    G1JointIndex.LeftWristRoll,       # 19
    G1JointIndex.LeftWristPitch,      # 20
    G1JointIndex.LeftWristYaw,        # 21
]

RIGHT_ARM_INDICES = [
    G1JointIndex.RightShoulderPitch,
    G1JointIndex.RightShoulderRoll,
    G1JointIndex.RightShoulderYaw,
    G1JointIndex.RightElbow,
    G1JointIndex.RightWristRoll,
    G1JointIndex.RightWristPitch,
    G1JointIndex.RightWristYaw,
]


def _arm_indices(side: str):
    if side == "left":
        return LEFT_ARM_INDICES
    if side == "right":
        return RIGHT_ARM_INDICES
    raise ValueError(f"side must be 'left' or 'right', got {side!r}")


class G1ArmController:
    """Programmatic interface to the G1 arms via the existing LiveTuner.

    All joint values are absolute radians, in myArm.G1Arm joint ordering.
    """

    def __init__(self, max_arm_speed_dps: float = None,
                 max_waist_speed_dps: float = None,
                 kp_arm: float = None,
                 kd_arm: float = None,
                 channel_id: int = 0,
                 gravity_comp=None):
        self._channel_id = channel_id
        self._dds_initialized = False
        self._connected = False

        # Locked-joint indices, per side, in this controller's own arm
        # ordering (0..6 for one arm). Tracking-error abort skips these
        # indices because by definition a locked motor can't track — but
        # the check stays armed for every other joint.
        # Public attribute: set by the choreograph caller after it has
        # decided which joint(s) are locked. Empty = no joints locked.
        self.locked_left: list[int] = []
        self.locked_right: list[int] = []

        # Optional speed overrides applied to the module globals so the
        # underlying slew uses our chosen rates.
        if max_arm_speed_dps is not None:
            glt.MAX_ARM_DEG_PER_SEC = float(max_arm_speed_dps)
        if max_waist_speed_dps is not None:
            glt.MAX_WAIST_DEG_PER_SEC = float(max_waist_speed_dps)

        # Optional PD gain overrides for the arm joints. The control loop
        # reads these globals every cycle, so updating them here takes
        # effect immediately on connect.
        if kp_arm is not None:
            glt.KP_ARM = float(kp_arm)
        if kd_arm is not None:
            glt.KD_ARM = float(kd_arm)

        # Optional gravity compensation. Pass a GravityComp instance to
        # enable feedforward torque on the arm joints. The LiveTuner
        # reads its .scale every tick, so the scale can be ramped live
        # via gravity_comp.set_scale().
        self._gravity_comp = gravity_comp

        # NOTE: do NOT instantiate LiveTuner here. Its __init__ creates a
        # MotionSwitcherClient which opens a DDS channel and requires
        # ChannelFactoryInitialize to have already been called. We defer
        # construction to connect().
        self.tuner = None

    # -------- lifecycle --------

    def connect(self):
        """Initialize DDS, start the 100 Hz control loop, capture init pose."""
        if self._connected:
            return
        if not self._dds_initialized:
            ChannelFactoryInitialize(self._channel_id)
            self._dds_initialized = True
        if self.tuner is None:
            self.tuner = LiveTuner(gravity_comp=self._gravity_comp)
        self.tuner.init_dds()
        self.tuner.start()
        self._connected = True
        print(f"[G1ArmController] Connected. "
              f"arm_max={glt.MAX_ARM_DEG_PER_SEC} deg/s, "
              f"waist_max={glt.MAX_WAIST_DEG_PER_SEC} deg/s, "
              f"kp_arm={glt.KP_ARM}, kd_arm={glt.KD_ARM}.")

    def disconnect(self):
        """Release arm_sdk control. The control thread is left running so the
        last commanded posture is held; flag-zero on the unused-joint slot
        relinquishes authority on the locomotion side."""
        if not self._connected:
            return
        with self.tuner.lock:
            self.tuner.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 0.0
            self.tuner.low_cmd.crc = self.tuner.crc.Crc(self.tuner.low_cmd)
            self.tuner.pub.Write(self.tuner.low_cmd)
        time.sleep(0.1)
        self._connected = False
        print("[G1ArmController] Disconnected (arm_sdk released).")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disconnect()

    # -------- state queries --------

    def get_current_q(self, side: str = "left") -> np.ndarray:
        """Return the *measured* joint angles (rad).
        side="left"/"right" → 7-vec for that arm.
        side="both"         → 14-vec [left(7), right(7)]."""
        if not self._connected:
            raise RuntimeError("Call connect() first.")
        if side == "both":
            return self.get_both_arms_q()
        idx = _arm_indices(side)
        with self.tuner.lock:
            st = self.tuner.low_state
            if st is None:
                raise RuntimeError("No lowstate received yet.")
            return np.array([float(st.motor_state[j].q) for j in idx])

    def get_init_q(self, side: str = "left") -> np.ndarray:
        """Joint angles that were captured the moment connect() returned.
        Useful for sanity-checking that the robot started in a known pose."""
        idx = _arm_indices(side)
        return np.array([self.tuner.init_q[j] for j in idx])

    def get_both_arms_q(self) -> np.ndarray:
        """14-vector: [left(7), right(7)] of measured joint angles."""
        return np.concatenate([self.get_current_q("left"),
                               self.get_current_q("right")])

    def dump(self):
        """Print all arm+waist offsets from init (for tuning / debugging)."""
        self.tuner.dump_offsets()

    # -------- motion --------

    def move_arm_to(self, q_abs, side: str = "left", settle: float = 0.5):
        """Slew one arm to absolute joint configuration q_abs (7-vec, rad).
        Blocks until the slew interpolation finishes; settle adds a wait
        so the PD loop converges before returning. 0.5 s default is
        usually enough for 8 deg/s slow runs; bump higher if you see
        residual tracking error."""
        q_abs = np.asarray(q_abs, dtype=float)
        if q_abs.shape != (7,):
            raise ValueError(f"q_abs must be shape (7,), got {q_abs.shape}")
        idx = _arm_indices(side)
        target_off_deg = self._abs_to_offset_deg(idx, q_abs)
        self._slew_simultaneous(idx, target_off_deg)
        if settle > 0:
            time.sleep(settle)

    def move_both_arms_to(self, q_abs_14, settle: float = 0.05):
        """Slew both arms together to a 14-vec [left(7), right(7)] in rad."""
        q_abs_14 = np.asarray(q_abs_14, dtype=float)
        if q_abs_14.shape != (14,):
            raise ValueError(f"q_abs_14 must be shape (14,), got {q_abs_14.shape}")
        idx = LEFT_ARM_INDICES + RIGHT_ARM_INDICES
        target_off_deg = self._abs_to_offset_deg(idx, q_abs_14)
        self._slew_simultaneous(idx, target_off_deg)
        if settle > 0:
            time.sleep(settle)

    def _locked_indices_for(self, side: str) -> list[int]:
        """Return locked-joint indices in the trajectory's own coordinates.

        side='left' or 'right' → indices into the 7-vec for that arm.
        side='both'            → indices into the 14-vec [left(7), right(7)],
                                 so right indices are shifted by +7.
        """
        if side == "left":
            return list(self.locked_left)
        if side == "right":
            return list(self.locked_right)
        # both
        return list(self.locked_left) + [7 + j for j in self.locked_right]

    def execute_trajectory(self, waypoints, side: str = "left",
                           smooth: bool = True,
                           duration: float = None,
                           dwell: float = 0.0, verbose: bool = True,
                           max_tracking_err_deg: float = 30.0):
        """Execute a sequence of joint configurations.

        smooth=True (default): fit a cubic spline through the waypoints and
        stream the result at the control loop rate. Continuous velocity,
        no stop-and-go between waypoints. Recommended.

        smooth=False: legacy per-waypoint slew. Discrete stops at each
        waypoint, controlled by dwell.

        side='left'/'right'   -> each waypoint is a 7-vec
        side='both'           -> each waypoint is a 14-vec (smooth not yet
                                 supported for 'both'; falls back to slew)
        duration              -> total trajectory time in seconds. If None
                                 and smooth=True, auto-computed from the
                                 path length and current max_arm_speed_dps.
        max_tracking_err_deg  -> abort if any *unlocked* joint lags this far
                                 behind. Indices in self.locked_left /
                                 self.locked_right are skipped. None disables
                                 the check entirely.
        """
        if smooth:
            return self.stream_smooth_trajectory(
                waypoints, side=side, duration=duration, verbose=verbose,
                max_tracking_err_deg=max_tracking_err_deg)

        # Legacy slew (or dual-arm). One slew per waypoint, blocking.
        locked = set(self._locked_indices_for(side))
        for i, q in enumerate(waypoints):
            if side == "both":
                self.move_both_arms_to(q)
            else:
                self.move_arm_to(q, side=side)
            if dwell > 0:
                time.sleep(dwell)
            if max_tracking_err_deg is not None and side != "both":
                q_actual = self.get_current_q(side)
                err_deg = np.degrees(np.abs(q_actual - np.asarray(q)))
                # Mask out locked joints from the abort check.
                unlocked = [k for k in range(len(err_deg)) if k not in locked]
                if unlocked:
                    max_err = float(np.max(err_deg[unlocked]))
                    if max_err > max_tracking_err_deg:
                        bad_in_unlocked = int(np.argmax(err_deg[unlocked]))
                        bad = unlocked[bad_in_unlocked]
                        print(f"[G1ArmController] ABORT at waypoint "
                              f"{i+1}/{len(waypoints)}: joint {bad} lagging "
                              f"{err_deg[bad]:.1f} deg "
                              f"(limit {max_tracking_err_deg} deg).")
                        return False
            if verbose:
                print(f"[G1ArmController] waypoint {i+1}/{len(waypoints)} reached")
        return True

    def stream_smooth_trajectory(self, waypoints, side: str = "left",
                                 duration: float = None,
                                 verbose: bool = True,
                                 max_tracking_err_deg: float = 30.0):
        """Fit a cubic spline through `waypoints` (joint space) and stream
        the result at the control loop rate. Zero velocity at start and
        end. Velocity bounded by current MAX_ARM_DEG_PER_SEC.

        side='left' or 'right' -> waypoints are (N, 7), streams to one arm.
        side='both'            -> waypoints are (N, 14) = [left(7), right(7)],
                                  streams to both arms simultaneously over a
                                  single shared trajectory.

        Aborts if any *unlocked* joint lags more than max_tracking_err_deg.
        Joints in self.locked_left / self.locked_right are masked out of the
        check (a locked motor can't track, by definition). Pass None to
        disable the check entirely.

        Returns True on success, False if motors lagged beyond
        max_tracking_err_deg at any point during streaming.
        """
        from scipy.interpolate import CubicSpline

        waypoints = np.asarray(waypoints, dtype=float)
        expected_dim = 14 if side == "both" else 7
        if waypoints.ndim != 2 or waypoints.shape[1] != expected_dim:
            raise ValueError(
                f"waypoints must be (N, {expected_dim}) for side={side!r}, "
                f"got {waypoints.shape}")
        n_pts = len(waypoints)
        if n_pts == 0:
            return True
        if n_pts == 1:
            if side == "both":
                return self.move_both_arms_to(waypoints[0]) is not False
            return self.move_arm_to(waypoints[0], side=side) is not False

        n_joints = waypoints.shape[1]
        seg_dists = np.array([
            np.max(np.abs(waypoints[i + 1] - waypoints[i]))
            for i in range(n_pts - 1)
        ])
        total_dist = float(seg_dists.sum())
        if total_dist < 1e-6:
            return True

        max_speed_rps = math.radians(glt.MAX_ARM_DEG_PER_SEC)
        if duration is None:
            duration = total_dist / max_speed_rps

        knot_times, splines, peak_vel = self._build_spline(
            waypoints, seg_dists, total_dist, duration, n_joints)
        if peak_vel > max_speed_rps:
            duration *= peak_vel / max_speed_rps
            knot_times, splines, peak_vel = self._build_spline(
                waypoints, seg_dists, total_dist, duration, n_joints)

        # Joint indices to drive
        if side == "both":
            idx = LEFT_ARM_INDICES + RIGHT_ARM_INDICES
        else:
            idx = _arm_indices(side)

        # Precompute which entries of err_deg the abort check should look
        # at — the locked-joint indices are excluded because a locked
        # motor by definition can't track. Empty `unlocked` ⇒ all joints
        # are locked (nonsense, but defensive) ⇒ skip the abort check.
        locked_set = set(self._locked_indices_for(side))
        unlocked = [k for k in range(n_joints) if k not in locked_set]

        dt = self.tuner.dt
        n_steps = int(math.ceil(duration / dt))

        if verbose:
            print(f"[trajectory] Streaming {n_pts} waypoints over "
                  f"{duration:.2f}s ({n_steps} steps @ {int(1/dt)} Hz, peak "
                  f"{math.degrees(peak_vel):.1f} deg/s, side={side}, "
                  f"{n_joints}-DOF)")
            if locked_set:
                print(f"[trajectory] tracking abort skips locked joint(s) "
                      f"{sorted(locked_set)}")

        t0 = time.time()
        last_print_pct = -10
        for step in range(n_steps + 1):
            t = min(step * dt, duration)
            q_t = np.array([float(spl(t)) for spl in splines])

            with self.tuner.lock:
                for j_idx, q in zip(idx, q_t):
                    self.tuner.target_q[j_idx] = q

            if (max_tracking_err_deg is not None
                    and unlocked
                    and step > 0 and step % 50 == 0):
                q_actual = np.array([
                    float(self.tuner.low_state.motor_state[j].q) for j in idx
                ])
                err_deg = np.degrees(np.abs(q_actual - q_t))
                err_unlocked = err_deg[unlocked]
                max_err = float(np.max(err_unlocked))
                if max_err > max_tracking_err_deg:
                    bad = unlocked[int(np.argmax(err_unlocked))]
                    print(f"[trajectory] WARN at t={t:.2f}s: joint {bad} "
                        f"lagging {err_deg[bad]:.1f} deg "
                        f"(threshold {max_tracking_err_deg}). Continuing.")

            if verbose:
                pct = int(100 * t / duration)
                if pct >= last_print_pct + 25:
                    print(f"[trajectory] ... {pct}%")
                    last_print_pct = pct

            next_t = t0 + (step + 1) * dt
            now = time.time()
            if now < next_t:
                time.sleep(next_t - now)

        if max_tracking_err_deg is not None and unlocked:
            # Small settle pad: the spline ends with zero commanded velocity
            # but the PD takes ~0.15-0.25s at KP=120 to actually converge
            # to the target after a high-speed motion. Without this pad,
            # the end-of-stream check fires during convergence and reports
            # spurious "ended with X deg lag" warnings. 0.20s is empirical.
            time.sleep(0.20)
            if side == "both":
                q_actual = self.get_both_arms_q()
            else:
                q_actual = self.get_current_q(side)
            err_deg = np.degrees(np.abs(q_actual - waypoints[-1]))
            err_unlocked = err_deg[unlocked]
            max_err = float(np.max(err_unlocked))
            if max_err > max_tracking_err_deg:
                bad = unlocked[int(np.argmax(err_unlocked))]
                print(f"[trajectory] WARN: ended with {err_deg[bad]:.1f} deg "
                      f"tracking error on joint {bad}")
        if verbose:
            print("[trajectory] Done.")
        return True

    def _build_spline(self, waypoints, seg_dists, total_dist, duration,
                      n_joints=7):
        """Build per-joint clamped cubic spline and report peak velocity."""
        from scipy.interpolate import CubicSpline
        knot_times = (np.concatenate([[0.0], np.cumsum(seg_dists)])
                      / total_dist * duration)
        splines = [
            CubicSpline(knot_times, waypoints[:, j],
                        bc_type=((1, 0.0), (1, 0.0)))
            for j in range(n_joints)
        ]
        t_samples = np.linspace(0, duration, 200)
        peak = 0.0
        for spl in splines:
            v = np.abs(spl(t_samples, 1))
            peak = max(peak, float(np.max(v)))
        return knot_times, splines, peak


    def home(self, smooth: bool = True, max_tracking_err_deg: float = 30.0):
        """Return both arms to their captured init pose (typically Get
        Ready / hanging).

        smooth=True (default): cubic-spline trajectory through the same
        streaming pipeline as execute_trajectory. Speed-capped, safe.
        smooth=False: legacy snap-to-init (the original tuner.home(),
        which sets target_q to init_q in one tick -- VIOLENT, do not use
        on real hardware).

        Note: this only homes the arms (left + right). The waist is left
        wherever it is. If you want the original behavior, call
        self.tuner.home() directly.
        """
        if not smooth:
            self.tuner.home()
            return

        # Build a 2-waypoint "trajectory": current -> init for both arms.
        q_init_l = np.array([self.tuner.init_q[j] for j in LEFT_ARM_INDICES])
        q_init_r = np.array([self.tuner.init_q[j] for j in RIGHT_ARM_INDICES])
        q_init_14 = np.concatenate([q_init_l, q_init_r])
        q_now_14 = self.get_both_arms_q()

        # If we're essentially at init already, nothing to do
        if np.max(np.abs(q_now_14 - q_init_14)) < math.radians(0.5):
            return

        waypoints = np.array([q_now_14, q_init_14])
        print("[home] Smoothly returning both arms to init pose...")
        return self.stream_smooth_trajectory(
            waypoints, side="both",
            max_tracking_err_deg=max_tracking_err_deg)

    # -------- internals --------

    def _abs_to_offset_deg(self, indices, q_abs):
        """Convert absolute target rad -> 'offset from init' deg, per joint."""
        out = []
        for j, q in zip(indices, q_abs):
            off_rad = float(q) - float(self.tuner.init_q[j])
            out.append(off_rad * 180.0 / math.pi)
        return out

    def _slew_simultaneous(self, indices, target_offsets_deg):
        """Linear-interpolate every listed joint from its current target
        offset to the new offset, capped by the per-joint max rate so the
        slowest joint sets the duration. Mirrors LiveTuner.apply_offset_set."""
        with self.tuner.lock:
            start_deg = []
            for j in indices:
                base = self.tuner.init_q[j]
                cur = self.tuner.target_q.get(j, base)
                start_deg.append((cur - base) * 180.0 / math.pi)

        max_steps = 1
        for j, sd, td in zip(indices, start_deg, target_offsets_deg):
            delta = abs(td - sd)
            if delta < 1e-6:
                continue
            rate = (glt.MAX_WAIST_DEG_PER_SEC
                    if j in WAIST_JOINTS else glt.MAX_ARM_DEG_PER_SEC)
            steps = int(math.ceil(delta / (rate * self.tuner.dt)))
            if steps > max_steps:
                max_steps = steps

        for k in range(max_steps):
            s = float(k + 1) / float(max_steps)
            with self.tuner.lock:
                for j, sd, td in zip(indices, start_deg, target_offsets_deg):
                    new_off_deg = sd + (td - sd) * s
                    self.tuner.target_q[j] = self.tuner.init_q[j] + d2r(new_off_deg)
            time.sleep(self.tuner.dt)


# ---- standalone smoke test ----
if __name__ == "__main__":
    print("Hold the arm or suspend the robot before running this.")
    input("Press Enter when ready... ")
    with G1ArmController(max_arm_speed_dps=10.0) as ctrl:
        print("Init q (left): ", ctrl.get_init_q("left"))
        print("Now q  (left): ", ctrl.get_current_q("left"))
        # Tiny move: shoulder pitch +5 deg from current
        q = ctrl.get_current_q("left")
        q[0] += math.radians(5)
        input("Press Enter to move LSP +5 deg from current... ")
        ctrl.move_arm_to(q, side="left")
        time.sleep(1.0)
        input("Press Enter to home... ")
        ctrl.home()
        time.sleep(1.0)