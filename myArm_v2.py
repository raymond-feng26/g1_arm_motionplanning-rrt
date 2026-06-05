#!/usr/bin/env python3
"""
myArm_v2.py

URDF-loaded drop-in replacement for the hand-coded G1Arm in myArm.py.

Identical public interface as myArm.G1Arm (same method names, same returns)
so existing code (g1_arm_controller, g1_ik_bridge, g1_plan_run,
g1_calibrate, rrt_planner) keeps working without changes.

Key differences from the original:
  - Chain is loaded from the full Unitree G1 URDF instead of being typed out
  - Joint limits read straight from the URDF
  - Supports both 'left' and 'right' arms via the side= argument
  - Adds a virtual fingertip link 12 cm past the palm so the last collision
    capsule conservatively covers the dex3 hand and its fingers

New in this revision:
  - inverse_kinematics_dls now accepts target_orientation (3x3 rotation)
    as a soft constraint. Use orientation_weight to scale it.
  - get_end_effector_pose() returns the full 4x4 fingertip transform.
  - G1Arm(..., locked_joints={6: angle}) holds the given joint(s) at a
    fixed value through IK and RRT sampling. Use this for broken motors
    (e.g. right wrist yaw): set angle to the current encoder reading.

Drop-in usage:
    from myArm_v2 import G1Arm          # instead of `from myArm import G1Arm`
    arm = G1Arm()                       # left arm by default
    arm = G1Arm(side='right')           # right arm
    arm = G1Arm(urdf_path='/path/to/g1.urdf')  # custom URDF location

If the URDF isn't next to this file, set env var G1_URDF_PATH.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ikpy.chain import Chain
from ikpy.link import URDFLink


# Default URDF path: look for g1_29dof_with_hand.urdf alongside this file,
# or accept an override via env var.
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URDF_PATH = os.environ.get(
    "G1_URDF_PATH",
    os.path.join(_HERE, "g1_29dof_with_hand.urdf"))

# Distance past the palm origin to a virtual "fingertip" tip, used to make
# the last collision capsule cover the dex3 hand. 12 cm is conservative for
# fingers in closed / neutral posture; bump if you plan with fingers spread.
FINGERTIP_OFFSET_M = 0.13

# "Hand bulk" capsule: a second hand capsule modeling the closed dex3 hand,
# which bulges toward the PALM side — base -y for the left hand and +y for the
# right at the grasp pose (palms face inward toward the box). It is NOT a chain
# link (that would move the IK end-effector); it is computed from the hand's FK
# frame in _capsule_segments, so it rides the real wrist orientation. Defined in
# the hand frame, which is base-aligned at the identity grasp orientation:
#   center = wrist + R_hand @ [FWD, in_sign*IN, 0]   (in_sign: left -1, right +1)
# and the capsule runs +/-HALFLEN along the hand x-axis through that center.
# Offsets per the measured closed hand; tune on the preview / hardware.
PALM_BULK_FWD     = 0.08   # center, along hand +x (toward fingertips) [m]
PALM_BULK_IN      = 0.035   # center, toward the palm (inward)          [m]
PALM_BULK_HALFLEN = 0.01    # half-length of the bulk capsule along +x  [m]

# Capsule radii, one per capsule in _capsule_segments order: the 7 chain
# segments (shoulder -> wrist->fingertip) then the palm-side hand-bulk capsule.
# Entry 6 (wrist->fingertip) stays thin so the on-axis rod never overlaps the
# box during grasp; entry 7 is the fatter hand-bulk capsule.
DEFAULT_LINK_RADII = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.04]


# ----- rotation helpers -----

def _orientation_error(R_a, R_b):
    """3-vec axis-angle rotation that takes R_a to R_b.

    Magnitude is the rotation angle in radians; direction is the axis.
    Handles the near-pi singular case via the eigenvector-of-(R+I) approach.
    """
    R_err = np.asarray(R_b) @ np.asarray(R_a).T
    cos_t = (np.trace(R_err) - 1.0) / 2.0
    cos_t = float(np.clip(cos_t, -1.0, 1.0))
    theta = float(np.arccos(cos_t))
    if theta < 1e-9:
        return np.zeros(3)
    if abs(theta - np.pi) < 1e-6:
        # Standard axis formula has sin(theta) -> 0 here. For a pi-rotation,
        # R = 2 a a^T - I, so (R + I)/2 = a a^T. Pick the column with the
        # largest diagonal entry of (R + I)/2 and normalize.
        M = (R_err + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(M)))
        axis = M[:, k] / np.sqrt(max(M[k, k], 1e-12))
        return axis * theta
    sin_t = np.sin(theta)
    axis = np.array([R_err[2, 1] - R_err[1, 2],
                     R_err[0, 2] - R_err[2, 0],
                     R_err[1, 0] - R_err[0, 1]]) / (2.0 * sin_t)
    return axis * theta


def rpy_to_R(roll, pitch, yaw):
    """ROS-convention roll-pitch-yaw (intrinsic XYZ = extrinsic ZYX) to
    3x3 rotation matrix. Angles in radians.
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# Path through the URDF tree for each arm. We only need the *start* (torso)
# and intermediate links so ikpy's auto-walk follows the right branch; we
# then trim past the palm and append the virtual fingertip ourselves.
def _arm_elements(side):
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    p = side  # "left" or "right"
    return [
        "torso_link",
        f"{p}_shoulder_pitch_joint",
        f"{p}_shoulder_pitch_link",
        f"{p}_shoulder_roll_joint",
        f"{p}_shoulder_roll_link",
        f"{p}_shoulder_yaw_joint",
        f"{p}_shoulder_yaw_link",
        f"{p}_elbow_joint",
        f"{p}_elbow_link",
        f"{p}_wrist_roll_joint",
        f"{p}_wrist_roll_link",
        f"{p}_wrist_pitch_joint",
        f"{p}_wrist_pitch_link",
        f"{p}_wrist_yaw_joint",
        f"{p}_wrist_yaw_link",
        f"{p}_hand_palm_joint",
        f"{p}_hand_palm_link",
    ]


def _build_chain(urdf_path, side, fingertip_offset_m):
    """Load the URDF, walk the requested arm branch, trim past the palm,
    fix the active mask, and append a virtual fingertip link.

    Returns (chain, active_joint_indices)
    The active_joint_indices are 7 ints, in this order:
      shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
      wrist_roll, wrist_pitch, wrist_yaw
    """
    import warnings
    elements = _arm_elements(side)
    # ikpy warns about "fixed link marked active" for the base and palm
    # joint during from_urdf_file. We correct the mask immediately after,
    # so suppress the noisy warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*is of type 'fixed' but set as active.*")
        chain = Chain.from_urdf_file(
            urdf_path,
            base_elements=elements,
            base_element_type="link",
            symbolic=False,
        )

    # ikpy auto-walks past palm into the fingers; trim back to the palm joint.
    palm_joint_name = f"{side}_hand_palm_joint"
    palm_idx = next(
        (i for i, l in enumerate(chain.links)
         if getattr(l, "name", "") == palm_joint_name),
        None)
    if palm_idx is None:
        raise RuntimeError(
            f"Could not find {palm_joint_name} in the loaded chain.")
    chain.links = chain.links[:palm_idx + 1]
    chain.active_links_mask = list(chain.active_links_mask[:palm_idx + 1])

    # Mark the base link and the fixed palm joint as inactive so IK doesn't
    # try to "solve" for them.
    for i, link in enumerate(chain.links):
        nm = getattr(link, "name", "")
        if "palm_joint" in nm or nm == "Base link":
            chain.active_links_mask[i] = False

    # Append a virtual fingertip link past the palm so the last collision
    # capsule covers the hand body.
    fingertip = URDFLink(
        name=f"{side}_fingertip_virtual",
        origin_translation=[fingertip_offset_m, 0.0, 0.0],
        origin_orientation=[0.0, 0.0, 0.0],
        rotation=[0.0, 0.0, 0.0],  # fixed
    )
    chain.links.append(fingertip)
    chain.active_links_mask.append(False)

    # Indices of the 7 active (revolute) joints
    active_joint_indices = [
        i for i, active in enumerate(chain.active_links_mask) if active
    ]
    if len(active_joint_indices) != 7:
        raise RuntimeError(
            f"Expected 7 active joints, got {len(active_joint_indices)}: "
            f"{[chain.links[i].name for i in active_joint_indices]}")

    return chain, active_joint_indices


class G1Arm:
    """G1 left or right arm. Same interface as the original myArm.G1Arm."""

    def __init__(self, side="left",
                 urdf_path=DEFAULT_URDF_PATH,
                 fingertip_offset_m=FINGERTIP_OFFSET_M,
                 link_radii=None,
                 locked_joints=None):
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(
                f"URDF not found at {urdf_path}. Set G1_URDF_PATH env var "
                f"or pass urdf_path=... to G1Arm.")
        self.side = side
        self.urdf_path = urdf_path
        self.arm, self.active_joint_indices = _build_chain(
            urdf_path, side, fingertip_offset_m)
        self.num_joints = 7

        # Joint limits from URDF, in active-joint order
        self.joint_limits = np.array([
            [self.arm.links[i].bounds[0], self.arm.links[i].bounds[1]]
            for i in self.active_joint_indices
        ], dtype=float)

        # Collision capsule radii — one per capsule (see _CAP_NAMES).
        self.link_radii = list(link_radii) if link_radii is not None \
            else list(DEFAULT_LINK_RADII)
        if len(self.link_radii) != 8:
            raise ValueError(
                f"link_radii must have 8 entries, got {len(self.link_radii)}.")

        # Palm-side hand-bulk capsule offsets toward the palm: left hand's palm
        # faces -y, right hand's +y (at the grasp pose), so the inward sign flips.
        self._palm_in_sign = -1.0 if side == "left" else +1.0

        # Obstacle list: each row = [cx, cy, cz, 0, 0, 0, sx, sy, sz].
        # Caller fills this in. Empty by default.
        self.obstacles = np.empty((0, 9))

        # Locked joints: {joint_idx: locked_angle_rad}. Used for broken
        # motors. The joint is held at locked_angle in IK; RRT samples it
        # to the same value via the clamped joint_limits below. FK and
        # collision-checking are unchanged -- the joint is still part of
        # the chain, just immobile.
        self.locked_joints = dict(locked_joints or {})
        for j, ang in self.locked_joints.items():
            if not (0 <= j < self.num_joints):
                raise ValueError(
                    f"locked joint index {j} out of range "
                    f"[0, {self.num_joints})")
            # Clamp limits to a zero-width interval so the uniform RRT
            # sampler can only pick the locked angle.
            self.joint_limits[j] = [ang, ang]

        # Precompute the constant per-link origin transforms for fast FK.
        # See _build_fast_fk for the bit-identity argument.
        self._build_fast_fk()

    # ----- fast forward kinematics (Lever 1) -----

    def _build_fast_fk(self):
        """Precompute the constant origin transform of every chain link so
        FK becomes (cached constant) @ (one axis rotation per joint), instead
        of ikpy rebuilding the full translation+rpy+axis product each call.

        Each ikpy link frame is  T_link(theta) = (Ttrans @ Trpy) @ Taxis(theta).
        Only Taxis depends on theta. We grab the constant (Ttrans @ Trpy) by
        calling ikpy's own get_link_frame_matrix(0.0): at theta=0, Taxis is the
        identity, so the call returns exactly (Ttrans @ Trpy). Using ikpy's own
        output (rather than re-deriving rpy) is what makes fast FK bit-identical
        to ikpy. At runtime we rebuild only Taxis via ikpy's axis_rotation_matrix
        (same formula -> same bits) and chain-multiply in the same left-to-right
        order ikpy uses, so every intermediate transform matches to the ULP.
        """
        from ikpy.utils.geometry import axis_rotation_matrix
        self._axis_rotation_matrix = axis_rotation_matrix
        self._fk_const = []
        self._fk_axis = []
        for link in self.arm.links:
            try:
                const = np.asarray(link.get_link_frame_matrix(0.0), dtype=float)
            except TypeError:
                # OriginLink / links that take no parameter -> identity origin
                const = np.eye(4)
            self._fk_const.append(const.copy())
            if getattr(link, "has_rotation", False):
                self._fk_axis.append(np.asarray(link.rotation, dtype=float))
            else:
                self._fk_axis.append(None)
        self._fast_fk_enabled = True

    def _fast_link_transforms(self, full_joints):
        """Cumulative 4x4 transform up to and including each link.
        Equivalent to ikpy's forward_kinematics(full_joints, full_kinematics=True).
        """
        T = np.eye(4)
        out = []
        for i, const in enumerate(self._fk_const):
            ax = self._fk_axis[i]
            if ax is None:
                T = T @ const
            else:
                R = self._axis_rotation_matrix(ax, full_joints[i])  # 3x3
                M = np.eye(4)
                M[:3, :3] = R
                T = T @ (const @ M)
            out.append(T.copy())
        return out

    # ----- joint vector conversions -----

    def _to_full_joints(self, joint_angles):
        """Expand 7-vec of active joint angles into the full chain vector."""
        full = np.zeros(len(self.arm.links))
        for i, angle in zip(self.active_joint_indices, joint_angles):
            full[i] = angle
        return full

    def _from_full_joints(self, full_joints):
        """Pick the 7 active joint angles out of a full chain vector."""
        return np.array([full_joints[i] for i in self.active_joint_indices])

    # ----- forward / inverse kinematics -----

    def forward_kinematics(self, joint_angles):
        """Return list of 4x4 transforms, one per chain link."""
        full = self._to_full_joints(joint_angles)
        if getattr(self, "_fast_fk_enabled", False):
            return self._fast_link_transforms(full)
        return self.arm.forward_kinematics(full, full_kinematics=True)

    def _geometric_jacobian(self, joint_angles, transforms=None):
        """Analytical 6x7 geometric (spatial) Jacobian at q, in world frame.

        Rows 0..2 are the linear (position) part dp/dq for each joint.
        Rows 3..5 are the angular (orientation) part w/dq, which is just
        the joint axis in world frame for revolute joints.

        Derivation (revolute joint i):
            J_linear[:, i]  = z_i (cross) (p_ee - p_i)
            J_angular[:, i] = z_i
        where z_i is the joint axis in world frame and p_i is the joint
        origin in world frame, both extracted from the FK transforms.

        Uses ONE forward_kinematics(q, full_kinematics=True) call. The
        old numerical Jacobian needed 14 separate FK calls per IK
        iteration (7 joints * central differences), so this is ~14x
        fewer FK evaluations. Also numerically exact instead of
        central-difference approximation, so behavior near singularities
        is cleaner.

        `transforms`: optional precomputed FK transform list (the output of
        forward_kinematics(q)). The DLS loop already computes FK to get the
        EE pose, so it passes that list in here to avoid a second identical
        FK call per iteration. When None, FK is computed as before, so the
        result is unchanged whether or not transforms are supplied.
        """
        T = transforms if transforms is not None \
            else self.forward_kinematics(joint_angles)

        # End-effector point is the last transform's origin (the virtual
        # fingertip link, matching get_end_effector_pose's reference).
        p_ee = T[-1][:3, 3]

        J = np.zeros((6, self.num_joints))
        for col, idx in enumerate(self.active_joint_indices):
            T_i = T[idx]
            # Joint axis in world frame: the link's local rotation axis
            # (URDF <axis>) rotated by the link's world rotation.
            axis_local = self.arm.links[idx].rotation
            z_i = T_i[:3, :3] @ axis_local
            # Joint origin in world frame: the link transform's translation.
            p_i = T_i[:3, 3]
            J[:3, col] = np.cross(z_i, p_ee - p_i)   # linear part
            J[3:, col] = z_i                         # angular part
        return J

    def inverse_kinematics(self, target_position, initial_position=None,
                           target_orientation=None, **dls_kwargs):
        """Solve IK for a 3D target position (and optional orientation).

        target_orientation: 3x3 rotation matrix in the same (torso) frame
        as target_position. When provided, IK adds it as a soft task with
        weight `orientation_weight`. Default None = position-only IK
        (preserves the original behavior).

        When `initial_position` (7-vec seed) is given, uses damped least
        squares iterating locally from the seed. This minimizes joint
        motion from the seed -- avoids ikpy's tendency to find wildly
        contorted solutions like 75-deg shoulder-yaw flips.

        Without a seed, falls back to ikpy's global solver (position only;
        passing target_orientation in this mode is an error -- give a seed).

        Extra kwargs are passed to `inverse_kinematics_dls`.
        """
        if initial_position is not None:
            return self.inverse_kinematics_dls(
                target_position, initial_position,
                target_orientation=target_orientation, **dls_kwargs)
        if target_orientation is not None:
            raise ValueError(
                "target_orientation requires an initial_position seed.")
        return self.inverse_kinematics_ikpy(target_position)

    def inverse_kinematics_ikpy(self, target_position):
        """Original ikpy-based IK. No seed, no joint-motion minimization."""
        target_frame = np.eye(4)
        target_frame[:3, 3] = np.asarray(target_position, dtype=float)
        full = self.arm.inverse_kinematics_frame(target_frame)
        return self._from_full_joints(full)

    def inverse_kinematics_dls(self, target_position, initial_position,
                               target_orientation=None,
                               orientation_weight=0.5,
                               reference_pose=None, null_space_weight=0.3,
                               max_iters=100, tol=1e-4, damping=0.05,
                               step_size=0.5):
        """Damped least squares IK with optional orientation task and
        null-space biasing.

        Primary task is always position (3D). If target_orientation (3x3)
        is given, orientation is added as a soft task with error stacked
        as [pos_err, orientation_weight * ori_err]. Position error is in
        meters; orientation error is axis-angle in radians.

        orientation_weight scales the orientation component relative to
        position. 0 reduces to position-only. Values around 0.3-1.0 are
        usually sensible; lower = orientation is more flexible.

        Null-space task: drive the joints toward `reference_pose` (defaults
        to `initial_position`). With a 7-DOF arm and a 3D task the null
        space is 4D; with a 6D task the null space is 1D. Set
        null_space_weight=0 to recover plain DLS.

        Locked joints (set on G1Arm) are held at their locked angles.
        Their Jacobian columns are zeroed, dq entries are zeroed, and the
        final clip to joint_limits enforces the value (since lo == hi for
        a locked joint).
        """
        q = np.array(initial_position, dtype=float)
        # Enforce locked joints at the start so FK / error are consistent.
        for j, ang in self.locked_joints.items():
            q[j] = ang

        ref = np.array(reference_pose if reference_pose is not None
                       else initial_position, dtype=float)
        # Never bias a locked joint away from its value.
        for j, ang in self.locked_joints.items():
            ref[j] = ang

        target_pos = np.asarray(target_position, dtype=float)
        R_target = (np.asarray(target_orientation, dtype=float)
                    if target_orientation is not None else None)
        use_ori = R_target is not None

        lo, hi = self.joint_limits[:, 0], self.joint_limits[:, 1]
        best_q, best_err = q.copy(), np.inf

        free_mask = np.ones(self.num_joints, dtype=bool)
        for j in self.locked_joints:
            free_mask[j] = False

        n_task = 6 if use_ori else 3

        for _ in range(max_iters):
            # One FK call per iteration: the transform list gives both the
            # EE pose (last transform) and the Jacobian inputs, so we pass
            # it into _geometric_jacobian instead of recomputing FK there.
            T = self.forward_kinematics(q)
            T0 = T[-1]
            p0 = T0[:3, 3]
            R0 = T0[:3, :3]

            err_pos = target_pos - p0
            if use_ori:
                err_ori = _orientation_error(R0, R_target)
                err = np.concatenate(
                    [err_pos, orientation_weight * err_ori])
            else:
                err = err_pos

            err_norm = np.linalg.norm(err)
            if err_norm < best_err:
                best_err, best_q = err_norm, q.copy()
            if err_norm < tol:
                break

            # Analytical geometric Jacobian: one FK call instead of 14.
            # Returns 6x7; the top 3 rows are dp/dq, the bottom 3 are dw/dq.
            # For position-only IK (n_task=3) we just slice off the angular
            # rows. For position+orientation we keep both, scaling the
            # angular block by orientation_weight to match the stacked
            # error vector.
            J_full = self._geometric_jacobian(q, transforms=T)
            if use_ori:
                J = np.vstack([J_full[:3, :],
                               orientation_weight * J_full[3:, :]])
            else:
                J = J_full[:3, :]

            # Zero out locked-joint columns so they contribute no motion.
            J[:, ~free_mask] = 0.0

            # Primary task: damped Jacobian-transpose step toward target.
            JJt = J @ J.T + damping ** 2 * np.eye(n_task)
            dq_task = J.T @ np.linalg.solve(JJt, err)

            # Null-space biasing: project (ref - q) into the Jacobian's
            # null space so it doesn't fight the primary task.
            if null_space_weight > 0:
                J_pinv = J.T @ np.linalg.solve(JJt, np.eye(n_task))
                N = np.eye(self.num_joints) - J_pinv @ J
                dq_null = N @ (ref - q)
                dq = dq_task + null_space_weight * dq_null
            else:
                dq = dq_task

            # Enforce locked joints (paranoia: null-space term can smear
            # into locked indices even though ref-q is zero there, due to
            # the projector coupling).
            dq[~free_mask] = 0.0

            q_new = np.clip(q + step_size * dq, lo, hi)
            if np.allclose(q_new, q, atol=1e-7):
                break
            q = q_new

        return best_q

    def get_end_effector_position(self, joint_angles):
        """Return XYZ of the (virtual fingertip) end-effector in torso frame."""
        if getattr(self, "_fast_fk_enabled", False):
            return self._fast_link_transforms(
                self._to_full_joints(joint_angles))[-1][:3, 3]
        full = self._to_full_joints(joint_angles)
        T = self.arm.forward_kinematics(full)
        return T[:3, 3]

    def get_end_effector_pose(self, joint_angles):
        """Return the full 4x4 fingertip transform in torso frame.
        T[:3, 3] is the position; T[:3, :3] is the orientation.
        The orientation is that of the wrist_yaw_link (the virtual
        fingertip link is fixed, so it inherits its parent's frame)."""
        if getattr(self, "_fast_fk_enabled", False):
            return self._fast_link_transforms(
                self._to_full_joints(joint_angles))[-1]
        full = self._to_full_joints(joint_angles)
        return self.arm.forward_kinematics(full)

    def get_link_position(self, joint_angles):
        """Return list of 3D origins for every link in the chain.
        Used by check_collision to build collision capsules."""
        transforms = self.forward_kinematics(joint_angles)
        return [T[:3, 3] for T in transforms]

    # ----- joint limits -----

    # Tolerance in radians used when checking whether a joint is within
    # its limits. Needed because linear interpolation between two configs
    # that both have q[j] == locked_angle produces ULP-level FP drift
    # (~1e-16 rad), which would otherwise trip the strict-inequality check
    # and break path shortcutting through locked joints. 1e-9 rad is
    # ~5.7e-8 degrees -- far below any meaningful angular precision (servo
    # precision is ~1e-3 rad) and far above FP ULP.
    JOINT_LIMIT_TOL = 1e-9

    def is_within_joint_limits(self, joint_angles):
        for i, q in enumerate(joint_angles):
            if (q < self.joint_limits[i, 0] - self.JOINT_LIMIT_TOL or
                    q > self.joint_limits[i, 1] + self.JOINT_LIMIT_TOL):
                return False
        return True

    # ----- collision check (Lever 2: closed-form slab test, batched) -----

    # Capsule layout: segments between these link indices (start, end).
    # Chain-based capsules (endpoints are link origins). The 8th capsule, the
    # palm-side hand bulk, is NOT here — it is computed from the hand FK frame
    # in _capsule_segments (see PALM_BULK_*). Chain indices after _build_chain:
    #   1 shoulder_pitch ... 7 wrist_yaw, 9 fingertip.
    _CAP_START = (1, 2, 3, 4, 5, 6, 7)
    _CAP_END   = (2, 3, 4, 5, 6, 7, 9)
    # Labels per capsule, aligned with _capsule_segments order (7 chain + bulk).
    # Used only by collision_report; check_collision never looks at these.
    _CAP_NAMES = ("shoulder pitch->roll", "shoulder roll->yaw",
                  "shoulder yaw->elbow",  "elbow->wrist roll",
                  "wrist roll->pitch",    "wrist pitch->yaw",
                  "wrist->hand",          "hand bulk (palm side)")
    # Hand FK frame used to place the bulk capsule: rotation from the fingertip
    # link (shares the wrist/palm orientation), anchored at the wrist-yaw origin.
    _HAND_ROT_LINK = 9
    _HAND_ANCHOR   = 7

    def check_collision(self, joint_angles):
        """True if any arm capsule intersects any obstacle box.

        Closed-form capsule-vs-AABB via the slab method, vectorized over all
        7 capsules x M obstacles in one broadcast (no Python per-pair loop).
        A capsule (segment, radius r) hits a box iff the segment intersects
        the box inflated by r on every side. Conservative at box corners
        (square instead of rounded): may report a hit a hair early near a
        corner, never misses one -- and unlike the old 20-sample scan it
        cannot tunnel through a thin obstacle.
        """
        if self.obstacles.shape[0] == 0:
            return False
        return bool(np.any(self._capsule_box_hits(joint_angles)))

    @staticmethod
    def _rpy_to_R_batch(rpy):
        """(M,3) roll/pitch/yaw -> (M,3,3) rotations, R = Rz@Ry@Rx (matches the
        module-level rpy_to_R, ROS convention). Columns of each R are the box's
        local axes expressed in the world/base frame."""
        r, p, y = rpy[:, 0], rpy[:, 1], rpy[:, 2]
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        M = rpy.shape[0]
        R = np.empty((M, 3, 3))
        R[:, 0, 0] = cy * cp
        R[:, 0, 1] = cy * sp * sr - sy * cr
        R[:, 0, 2] = cy * sp * cr + sy * sr
        R[:, 1, 0] = sy * cp
        R[:, 1, 1] = sy * sp * sr + cy * cr
        R[:, 1, 2] = sy * sp * cr - cy * sr
        R[:, 2, 0] = -sp
        R[:, 2, 1] = cp * sr
        R[:, 2, 2] = cp * cr
        return R

    def _box_rotations(self):
        """(M,3,3) world<-local rotations for the current obstacles, built from
        their (roll,pitch,yaw) slots. Cached on the obstacles array identity so
        the hot RRT path doesn't rebuild them every call (the world array is
        replaced, not mutated, when the scene changes)."""
        boxes = self.obstacles
        cache = getattr(self, "_obb_cache", None)
        if cache is not None and cache[0] is boxes:
            return cache[1]
        R_all = self._rpy_to_R_batch(boxes[:, 3:6])
        self._obb_cache = (boxes, R_all)
        return R_all

    def _capsule_segments(self, joint_angles):
        """(P1, P2, radii) for all 8 capsules: the 7 chain segments plus the
        computed palm-side hand-bulk capsule. One FK call. P1/P2 are (8,3),
        radii is (8,).

        The bulk capsule models the closed hand, which bulges toward the palm
        (left -y / right +y at the grasp pose). It is placed from the hand FK
        frame — center = wrist_anchor + R_hand @ [FWD, in_sign*IN, 0], extending
        +/-HALFLEN along the hand x-axis — so it rides the real wrist orientation
        rather than assuming a world direction. It is NOT a chain link, so the
        IK end-effector stays at the fingertip.
        """
        T = self.forward_kinematics(joint_angles)
        p = [t[:3, 3] for t in T]
        P1 = [p[i] for i in self._CAP_START]
        P2 = [p[i] for i in self._CAP_END]

        R_hand = T[self._HAND_ROT_LINK][:3, :3]
        anchor = p[self._HAND_ANCHOR]
        center = anchor + R_hand @ np.array(
            [PALM_BULK_FWD, self._palm_in_sign * PALM_BULK_IN, 0.0])
        axis = R_hand @ np.array([PALM_BULK_HALFLEN, 0.0, 0.0])
        P1.append(center - axis)
        P2.append(center + axis)
        return np.array(P1), np.array(P2), np.asarray(self.link_radii, float)

    def _capsule_box_hits(self, joint_angles):
        """(K, M) boolean array: entry [k, m] is True iff capsule k intersects
        obstacle m. K = 8 capsules (7 chain segments + the palm-side hand bulk,
        see _capsule_segments). Single source of the slab math, shared by
        check_collision (any-True) and collision_report (which pairs). One FK
        call. Caller guarantees self.obstacles is non-empty.

        Oriented (OBB): each capsule segment is transformed into the box's local
        frame (q = R^T (p - center)) and tested against the origin-centred box
        inflated by the capsule radius. When a box's (roll,pitch,yaw) are zero
        this is bit-identical to an axis-aligned slab test; non-zero yaw lets a
        tilted box (or table) be represented tightly instead of by a loose AABB.
        """
        P1, P2, radii = self._capsule_segments(joint_angles)  # (8,3),(8,3),(8,)
        R  = radii[:, None, None]                            # (K,1,1)

        boxes   = self.obstacles                            # (M,9)
        centers = boxes[:, 0:3]                             # (M,3)
        halves  = boxes[:, 6:9] / 2.0                       # (M,3)
        Rbox    = self._box_rotations()                     # (M,3,3) world<-local

        # Capsule endpoints in each box's local frame: q = Rbox^T (P - center).
        # einsum 'mji,kmj->kmi' contracts j, i.e. applies Rbox[m].T to (P-center).
        d1 = P1[:, None, :] - centers[None, :, :]           # (K,M,3)
        d2 = P2[:, None, :] - centers[None, :, :]           # (K,M,3)
        q1 = np.einsum('mji,kmj->kmi', Rbox, d1)            # (K,M,3) local
        q2 = np.einsum('mji,kmj->kmi', Rbox, d2)            # (K,M,3) local

        # Slab test against the origin-centred box, inflated by capsule radius.
        hi =  halves[None, :, :] + R                        # (K,M,3)
        lo = -hi
        d  = q2 - q1                                        # (K,M,3) local dir
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(np.abs(d) > 1e-12, 1.0 / d, np.inf)

        t1 = (lo - q1) * inv                                # (K,M,3)
        t2 = (hi - q1) * inv
        tmin = np.maximum(0.0, np.max(np.minimum(t1, t2), axis=2))  # (K,M)
        tmax = np.minimum(1.0, np.min(np.maximum(t1, t2), axis=2))  # (K,M)
        return tmin <= tmax

    def collision_report(self, joint_angles, obstacle_names=None):
        """List of (capsule_label, obstacle_label) for every colliding pair.

        Diagnostic companion to check_collision: instead of a single bool, it
        names which arm segment hits which obstacle. obstacle_names is an
        optional list parallel to self.obstacles (e.g. ["torso","box","table"]);
        if missing/short, falls back to "obstacle {m}". Returns [] when clear.
        """
        if self.obstacles.shape[0] == 0:
            return []
        hits = self._capsule_box_hits(joint_angles)
        out = []
        for k, m in np.argwhere(hits):
            cap = (self._CAP_NAMES[k] if k < len(self._CAP_NAMES)
                   else f"capsule {k}")
            if obstacle_names is not None and m < len(obstacle_names):
                obs = obstacle_names[m]
            else:
                obs = f"obstacle {m}"
            out.append((cap, obs))
        return out

    def check_capsule_box_collision(self, p1, p2, radius, box):
        """Single capsule-vs-OBB slab test. Same math as the batched
        check_collision, kept as a readable per-pair reference (and used by
        any caller that tests one capsule at a time). Honors the box's
        (roll,pitch,yaw) slots; reduces to the AABB test when they are zero."""
        p1 = np.asarray(p1, float)
        p2 = np.asarray(p2, float)
        center = box[0:3]
        half = box[6:9] / 2.0 + radius          # inflate by capsule radius
        Rb = rpy_to_R(box[3], box[4], box[5])
        q1 = Rb.T @ (p1 - center)               # endpoints in box-local frame
        q2 = Rb.T @ (p2 - center)
        lo, hi = -half, half
        d = q2 - q1
        inv = np.divide(1.0, d, out=np.full(3, np.inf),
                        where=np.abs(d) > 1e-12)
        t1 = (lo - q1) * inv
        t2 = (hi - q1) * inv
        tmin = max(0.0, float(np.max(np.minimum(t1, t2))))
        tmax = min(1.0, float(np.min(np.maximum(t1, t2))))
        return tmin <= tmax

    # ---- legacy sampled distance (superseded by the slab test above) ----
    # Kept only as a ground-truth/reference helper. NOT used by the live
    # collision path anymore. The 20-sample scan was both slow (it was the
    # dominant planning cost) and unsound (could miss a thin obstacle
    # between samples). Prefer check_collision / check_capsule_box_collision.

    def _capsule_box_sampled(self, p1, p2, radius, box):
        box_center = box[:3]
        box_half = box[6:9] / 2.0
        return self.closest_dist_on_segment(
            p1, p2, box_center - box_half, box_center + box_half) < radius

    def closest_dist_on_segment(self, p1, p2, box_min, box_max,
                                num_samples=20):
        """Minimum distance from any point on segment p1->p2 to the box.
        Legacy reference; superseded by the slab test."""
        ts = np.linspace(0.0, 1.0, num_samples)
        direction = np.asarray(p2) - np.asarray(p1)
        min_dist = float("inf")
        for t in ts:
            point = p1 + t * direction
            d = self.dist_point_to_box(point, box_min, box_max)
            if d < min_dist:
                min_dist = d
        return min_dist

    def dist_point_to_box(self, point, box_min, box_max):
        closest = np.clip(point, box_min, box_max)
        return np.linalg.norm(closest - point)

    # ----- planning helpers -----

    def sample_random_configuration(self):
        return np.random.uniform(
            self.joint_limits[:, 0],
            self.joint_limits[:, 1])

    # ----- visualization (live animation; for SSH use g1_plan_run.py's PNG/HTML) -----

    def visualize_path(self, path, obstacles=True, pause_time=0.5):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        def update(frame):
            ax.clear()
            full = self._to_full_joints(path[frame])
            self.arm.plot(full, ax)
            if obstacles:
                for obs in self.obstacles:
                    self._plot_box(ax, obs)
            ee_positions = [self.get_end_effector_position(q)
                            for q in path[:frame + 1]]
            if len(ee_positions) > 1:
                ee_arr = np.array(ee_positions)
                ax.plot(ee_arr[:, 0], ee_arr[:, 1], ee_arr[:, 2],
                        "b-", linewidth=2, alpha=0.5, label="EE path")
            start_ee = self.get_end_effector_position(path[0])
            goal_ee = self.get_end_effector_position(path[-1])
            ax.scatter(*start_ee, color="green", s=200, marker="o", label="Start")
            ax.scatter(*goal_ee, color="red", s=200, marker="*", label="Goal")
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            ax.set_xlim([-0.6, 0.6])
            ax.set_ylim([-0.4, 0.6] if self.side == "left" else [-0.6, 0.4])
            ax.set_zlim([-0.5, 0.6])
            ax.set_title(f"{self.side} arm  -  frame {frame + 1}/{len(path)}")
            ax.legend()

        # Keep a reference so the animation isn't GC'd before plt.show()
        self._anim = FuncAnimation(fig, update, frames=len(path),
                                   interval=pause_time * 1000, repeat=True)
        plt.show()

    def _plot_box(self, ax, box):
        center = box[:3]
        size = box[6:9]
        r = size / 2
        v = [
            center + [-r[0], -r[1], -r[2]], center + [+r[0], -r[1], -r[2]],
            center + [+r[0], +r[1], -r[2]], center + [-r[0], +r[1], -r[2]],
            center + [-r[0], -r[1], +r[2]], center + [+r[0], -r[1], +r[2]],
            center + [+r[0], +r[1], +r[2]], center + [-r[0], +r[1], +r[2]],
        ]
        edges = [[0, 1], [1, 2], [2, 3], [3, 0],
                 [4, 5], [5, 6], [6, 7], [7, 4],
                 [0, 4], [1, 5], [2, 6], [3, 7]]
        for a, b in edges:
            ax.plot3D(*zip(v[a], v[b]), "r-", alpha=0.6, linewidth=1)


# ----- smoke test -----
if __name__ == "__main__":
    print(f"URDF: {DEFAULT_URDF_PATH}")
    arm_l = G1Arm(side="left")
    arm_r = G1Arm(side="right")
    print(f"\nLeft  joint limits (deg):")
    for i, (lo, hi) in enumerate(arm_l.joint_limits):
        labels = ["LSP", "LSR", "LSY", "LE", "LWR", "LWP", "LWY"]
        print(f"  {labels[i]}: [{np.degrees(lo):+7.1f}, {np.degrees(hi):+7.1f}]")

    q_zero = np.zeros(7)
    print(f"\nLeft  EE at q=0:  {arm_l.get_end_effector_position(q_zero)}")
    print(f"Right EE at q=0:  {arm_r.get_end_effector_position(q_zero)}")

    # Sanity: clearance pose from last hardware reading
    q_clear = np.radians([16.5, 18.6, -0.7, 58.2, 4.4, 1.6, 1.8])
    ee = arm_l.get_end_effector_position(q_clear)
    print(f"\nLeft EE at clearance pose (deg):")
    print(f"  q  = {[round(d, 1) for d in np.degrees(q_clear)]}")
    print(f"  EE = {ee}")

    # ----- orientation IK smoke test -----
    print("\n--- Orientation IK smoke test ---")
    # Pick a target position and a target orientation (rotation matrix).
    # Use the current orientation at q_clear as a sanity check: solving
    # for (pos = ee + delta, ori = R_clear) from q_clear seed should give
    # a small joint motion and tiny errors.
    T_clear = arm_l.get_end_effector_pose(q_clear)
    R_clear = T_clear[:3, :3]
    target_pos = ee + np.array([0.05, 0.0, 0.02])

    print(f"Target position : {target_pos}")
    print(f"Target orientation : keep current R")
    q_sol = arm_l.inverse_kinematics(
        target_pos, initial_position=q_clear,
        target_orientation=R_clear, orientation_weight=0.5)
    T_sol = arm_l.get_end_effector_pose(q_sol)
    p_err = np.linalg.norm(T_sol[:3, 3] - target_pos)
    o_err = np.linalg.norm(_orientation_error(T_sol[:3, :3], R_clear))
    print(f"  pos err: {p_err*1000:.2f} mm")
    print(f"  ori err: {np.degrees(o_err):.2f} deg")
    print(f"  joint delta (deg): "
          f"{np.round(np.degrees(q_sol - q_clear), 2)}")

    # ----- locked-joint smoke test -----
    print("\n--- Locked-joint smoke test (right arm, lock RWY at 0.3 rad) ---")
    arm_r_locked = G1Arm(side="right", locked_joints={6: 0.3})
    print(f"Right RWY limits after lock : {arm_r_locked.joint_limits[6]}")
    q_seed = np.zeros(7)
    q_seed[6] = 0.3  # must agree with the lock or DLS will overwrite it
    target_r = arm_r_locked.get_end_effector_position(q_seed) \
                + np.array([0.0, 0.05, 0.05])
    q_sol = arm_r_locked.inverse_kinematics(
        target_r, initial_position=q_seed)
    print(f"  Target  : {target_r}")
    print(f"  Solved EE: {arm_r_locked.get_end_effector_position(q_sol)}")
    print(f"  q_sol[6]  (should equal 0.300): {q_sol[6]:.4f}")
    print(f"  sample x3 (q[6] should always = 0.300):")
    for _ in range(3):
        s = arm_r_locked.sample_random_configuration()
        print(f"    {s[6]:.4f}")