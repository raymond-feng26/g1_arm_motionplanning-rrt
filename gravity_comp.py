"""Gravity-compensation feedforward torque for the G1 arms+waist.

Computes per-joint gravity torque from the URDF using pinocchio and exposes
a single function that returns the torque vector for all controlled joints
in arm_sdk index order. Intended for use inside the LiveTuner control loop
at 100 Hz: ~6 us/call on a recent CPU, well under the loop budget.

Usage:
    from gravity_comp import GravityComp
    gc = GravityComp(urdf_path="g1_29dof_with_hand.urdf", scale=0.0)

    # In control loop, after measuring current arm/waist q:
    tau_ff = gc.compute(q_arm_sdk)  # dict {joint_idx: tau_nm}
    cmd.motor_cmd[j].tau = tau_ff[j] for j in CONTROLLED_JOINTS

The `scale` parameter is a 0..1 multiplier applied to the computed torque,
used to ramp gravity comp in safely. Start at 0.0, verify nothing breaks,
then raise to 0.5, 0.8, 1.0 with hardware observation at each step.
A sign-flipped value (-1.0) is sometimes needed if the arm_sdk firmware
expects torque in the opposite convention; flip via the `sign` argument
on init if hardware testing shows the arm sagging *more* with comp on.

The torque convention matches pinocchio: positive joint torque produces
positive joint velocity. Standard ROS / URDF convention.
"""
import numpy as np
import pinocchio as pin


# Joint index mapping: arm_sdk index (matches G1JointIndex in
# g1_arm_live_tuner.py) -> pinocchio joint name (must exist in the URDF).
# Only includes joints we apply feedforward to. Hand joints are excluded
# because they're on a separate DDS topic and the locomotion computer
# doesn't apply tau_ff to them via arm_sdk.
ARM_SDK_NAMES = {
    # Waist
    12: "waist_yaw_joint",
    13: "waist_roll_joint",
    14: "waist_pitch_joint",
    # Left arm
    15: "left_shoulder_pitch_joint",
    16: "left_shoulder_roll_joint",
    17: "left_shoulder_yaw_joint",
    18: "left_elbow_joint",
    19: "left_wrist_roll_joint",
    20: "left_wrist_pitch_joint",
    21: "left_wrist_yaw_joint",
    # Right arm
    22: "right_shoulder_pitch_joint",
    23: "right_shoulder_roll_joint",
    24: "right_shoulder_yaw_joint",
    25: "right_elbow_joint",
    26: "right_wrist_roll_joint",
    27: "right_wrist_pitch_joint",
    28: "right_wrist_yaw_joint",
}


class GravityComp:
    """Computes gravity-compensation feedforward torque per joint.

    On init: loads the URDF, builds the pinocchio model, builds the mapping
    from arm_sdk joint indices to pinocchio velocity indices. Allocates
    reusable state (data object, q vector) so compute() does no per-call
    allocation in steady state.

    The full pinocchio q-vector includes legs and hands — joints we don't
    control via arm_sdk. The locomotion computer holds the legs; the hands
    are on a separate channel. For computing arm/waist gravity comp, we
    treat the unobserved joints as fixed at zero. This is correct as long
    as the robot is standing (legs roughly straight, body upright). If the
    robot ever sits or kneels, the assumption breaks down for waist joints
    only; arm/waist gravity torques are still correct because those depend
    on arm+waist joints only, not on leg joints (the kinematic chain from
    floating base → torso → arm/waist links doesn't include legs).
    """

    def __init__(self, urdf_path, scale=0.0, sign=+1.0,
                 allow_joints=None, verbose=True):
        """allow_joints: iterable of arm_sdk indices to apply tau_ff to.
        Defaults to the 14 arm joints (sdk_idx 15..28). Waist joints
        (12, 13, 14) are EXCLUDED by default — they're balance-relevant
        and the locomotion controller may already feed-forward gravity
        for them; applying our value on top risks fighting that loop.
        Add waist joints to the allow_joints set only after hardware
        testing has confirmed they don't interact badly."""
        self.scale = float(scale)
        self.sign = float(sign)

        if allow_joints is None:
            allow_joints = set(range(15, 29))   # left + right arm only
        self.allow = set(int(j) for j in allow_joints)

        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.q = pin.neutral(self.model).copy()

        # Build arm_sdk_idx -> pinocchio v_idx, restricted to allow_joints
        name_to_vidx = {}
        for j_id in range(1, self.model.njoints):
            name_to_vidx[self.model.names[j_id]] = self.model.joints[j_id].idx_v

        self.sdk_to_pin = {}
        missing = []
        for sdk_idx, name in ARM_SDK_NAMES.items():
            if sdk_idx not in self.allow:
                continue
            if name in name_to_vidx:
                self.sdk_to_pin[sdk_idx] = name_to_vidx[name]
            else:
                missing.append(name)
        if missing:
            raise RuntimeError(
                f"URDF is missing expected joints: {missing}. "
                f"Check ARM_SDK_NAMES against the URDF.")

        if verbose:
            print(f"[gravity_comp] Loaded {urdf_path}")
            print(f"[gravity_comp] Total robot mass: "
                  f"{sum(I.mass for I in self.model.inertias[1:]):.2f} kg")
            excluded = set(ARM_SDK_NAMES) - self.allow
            print(f"[gravity_comp] Applying tau_ff to "
                  f"{sorted(self.sdk_to_pin.keys())}")
            if excluded:
                print(f"[gravity_comp] Excluded (will not receive tau_ff): "
                      f"{sorted(excluded)}")
            print(f"[gravity_comp] scale={self.scale:.2f}, "
                  f"sign={self.sign:+.1f}")

    def set_scale(self, scale):
        """Set the multiplier applied to computed torques. 0 = off, 1 = full."""
        self.scale = max(0.0, min(1.0, float(scale)))

    def compute(self, q_sdk):
        """Compute gravity-compensation tau_ff for all mapped arm_sdk joints.

        q_sdk: dict {arm_sdk_joint_idx: q_radians}, providing current
               positions for the joints we control. Missing joints stay
               at their last value (or initial 0).

        Returns: dict {arm_sdk_joint_idx: tau_nm}, signed and scaled,
                 ready to write into cmd.motor_cmd[j].tau directly.
        """
        # Update only the joints we have measurements for
        for sdk_idx, q_val in q_sdk.items():
            pin_idx = self.sdk_to_pin.get(sdk_idx)
            if pin_idx is not None:
                self.q[pin_idx] = q_val

        tau_full = pin.computeGeneralizedGravity(self.model, self.data, self.q)

        out = {}
        s = self.sign * self.scale
        for sdk_idx, pin_idx in self.sdk_to_pin.items():
            out[sdk_idx] = float(s * tau_full[pin_idx])
        return out

    def joint_indices(self):
        """Return the set of arm_sdk indices this comp produces values for.
        Useful for checking against CONTROLLED_JOINTS."""
        return set(self.sdk_to_pin.keys())