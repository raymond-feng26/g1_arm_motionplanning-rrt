#!/usr/bin/env python3
"""
myArm_dual.py

14-DOF dual-arm wrapper around two myArm_v2.G1Arm instances.

Public interface matches what rrt_planner.RRTConnect expects:
  - num_joints (int)
  - joint_limits (N, 2)
  - sample_random_configuration() -> N-vec
  - is_within_joint_limits(q) -> bool
  - check_collision(q) -> bool
  - get_end_effector_position(q) -> 3-vec  (returns midpoint of two EEs;
    for per-arm position use arm.left/.right or split())
  - get_link_position(q) -> list of 3D positions for visualization

Configuration vector layout (14,):
    q[0:7]  = left arm  [LSP, LSR, LSY, LE, LWR, LWP, LWY]
    q[7:14] = right arm [RSP, RSR, RSY, RE, RWR, RWP, RWY]

New in this revision:
  - inverse_kinematics(...) accepts per-arm orientation targets
    (target_left_ori, target_right_ori) as optional 3x3 rotation matrices.
    Pass None on either side to do position-only IK on that arm. This is
    the flag for "orientation control on left only" while the right RWY
    is broken: pass target_left_ori=R but leave target_right_ori=None.
  - G1DualArm(..., locked_joints_left=..., locked_joints_right=...)
    forwards to the per-arm G1Arm constructors. Use this to clamp the
    right-arm RWY to its current encoder reading at plan time:
        dual = G1DualArm(locked_joints_right={6: current_rwy_angle})

Usage:
    from myArm_dual import G1DualArm
    dual = G1DualArm()
    dual.obstacles = np.array([...])
    q = np.concatenate([q_left, q_right])
    dual.check_collision(q)

    # Position-only IK (unchanged behavior):
    q_goal = dual.inverse_kinematics(target_l, target_r, q_seed_14)

    # With left-arm orientation:
    q_goal = dual.inverse_kinematics(
        target_l, target_r, q_seed_14,
        target_left_ori=R_left,
        target_right_ori=None,
        orientation_weight=0.5)
"""

import numpy as np
from myArm_v2 import G1Arm


def _segment_segment_distance(p1, p2, p3, p4):
    """Minimum distance between two 3D line segments p1-p2 and p3-p4.
    Standard closest-points-on-segments algorithm.
    """
    p1, p2, p3, p4 = (np.asarray(x, dtype=float) for x in (p1, p2, p3, p4))
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)

    eps = 1e-9
    if a <= eps and e <= eps:
        # Both segments are degenerate points
        return float(np.linalg.norm(p1 - p3))
    if a <= eps:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e <= eps:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b
            if denom != 0.0:
                s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)
    closest1 = p1 + s * d1
    closest2 = p3 + t * d2
    return float(np.linalg.norm(closest1 - closest2))


class G1DualArm:
    """14-DOF dual arm. Wraps two G1Arm instances and adds inter-arm
    collision checking."""

    def __init__(self, urdf_path=None, fingertip_offset_m=0.13,
                 link_radii=None,
                 locked_joints_left=None,
                 locked_joints_right=None):
        kwargs = dict(fingertip_offset_m=fingertip_offset_m)
        if urdf_path is not None:
            kwargs["urdf_path"] = urdf_path
        if link_radii is not None:
            kwargs["link_radii"] = link_radii

        self.left = G1Arm(side="left",
                          locked_joints=locked_joints_left, **kwargs)
        self.right = G1Arm(side="right",
                           locked_joints=locked_joints_right, **kwargs)

        self.num_joints = 14
        self.joint_limits = np.vstack([self.left.joint_limits,
                                       self.right.joint_limits])

        # For visualization compatibility with save_path_html/png.
        # The full link-position list returned by get_link_position is
        # [left.links (10), right.links (10)]. Active joints are at
        # left.active_joint_indices and (right.active_joint_indices + 10).
        # Plus the two fingertip ends (last link of each arm = idx 9 and 19).
        n_left_links = len(self.left.arm.links)
        self.active_joint_indices = (
            list(self.left.active_joint_indices) + [n_left_links - 1]
            + [i + n_left_links for i in self.right.active_joint_indices]
            + [2 * n_left_links - 1]
        )
        # Stub `arm` attribute so save_path_html(...).arm.links works
        # for length queries. We return a dummy with `.links` of the
        # combined size.
        class _DualChain:
            def __init__(self, n):
                self.links = [None] * n
        self.arm = _DualChain(2 * n_left_links)

        # Shared obstacle list, written into both subarms each time check
        # is called (so they get the same view of the world).
        self.obstacles = np.empty((0, 9))

    # ----- joint vector helpers -----

    def split(self, q14):
        q14 = np.asarray(q14)
        return q14[:7], q14[7:14]

    def join(self, q_left, q_right):
        return np.concatenate([q_left, q_right])

    # ----- planning interface -----

    def sample_random_configuration(self):
        return np.random.uniform(self.joint_limits[:, 0],
                                 self.joint_limits[:, 1])

    def is_within_joint_limits(self, q14):
        # Match G1Arm's tolerance so FP drift through locked joints
        # doesn't trip the strict-inequality check.
        tol = G1Arm.JOINT_LIMIT_TOL
        for i, q in enumerate(q14):
            if (q < self.joint_limits[i, 0] - tol or
                    q > self.joint_limits[i, 1] + tol):
                return False
        return True

    def check_collision(self, q14):
        """True if either arm collides with obstacles OR with each other."""
        ql, qr = self.split(q14)

        # Push our obstacle list to each subarm so its check_collision sees
        # the same world.
        self.left.obstacles = self.obstacles
        self.right.obstacles = self.obstacles

        if self.left.check_collision(ql):
            return True
        if self.right.check_collision(qr):
            return True
        return self._check_inter_arm(ql, qr)

    def collision_report(self, q14, obstacle_names=None):
        """Per-pair collision diagnostics for both arms plus the inter-arm
        check. Returns a list of (label, obstacle) tuples: arm-vs-obstacle
        hits are prefixed "L "/"R "; an arm-vs-arm hit is one entry. Empty
        list when collision-free. obstacle_names is parallel to self.obstacles.
        """
        ql, qr = self.split(q14)
        self.left.obstacles = self.obstacles
        self.right.obstacles = self.obstacles
        out = []
        for cap, obs in self.left.collision_report(ql, obstacle_names):
            out.append((f"L {cap}", obs))
        for cap, obs in self.right.collision_report(qr, obstacle_names):
            out.append((f"R {cap}", obs))
        if self._check_inter_arm(ql, qr):
            out.append(("L arm", "R arm (inter-arm)"))
        return out

    def _arm_capsules(self, arm, q):
        """Return list of (start_xyz, end_xyz, radius) per capsule, including
        the computed palm-side hand-bulk capsule. Mirrors G1Arm._capsule_segments."""
        P1, P2, radii = arm._capsule_segments(q)
        return [(P1[i], P2[i], radii[i]) for i in range(len(radii))]

    def _check_inter_arm(self, q_left, q_right):
        """Check every left capsule vs every right capsule for intersection.
        Skip the first two segments per side (shoulder-only motion never
        approaches the other arm's shoulder) for performance."""
        left_caps = self._arm_capsules(self.left, q_left)
        right_caps = self._arm_capsules(self.right, q_right)
        # Skip shoulder pitch->roll and roll->yaw on both sides -- those
        # segments are <8 cm long and 0.20 m apart, never collide. Saves
        # 5/7 of the capsule-pair checks at no safety cost.
        for ls, le, lr in left_caps[2:]:
            for rs, re, rr in right_caps[2:]:
                d = _segment_segment_distance(ls, le, rs, re)
                if d < (lr + rr):
                    return True
        return False

    # ----- FK / EE helpers (for visualization) -----

    def get_end_effector_position(self, q14):
        """Return midpoint of the two fingertip positions. For per-arm
        position use self.left.get_end_effector_position(q14[:7]) directly."""
        ql, qr = self.split(q14)
        ee_l = self.left.get_end_effector_position(ql)
        ee_r = self.right.get_end_effector_position(qr)
        return 0.5 * (ee_l + ee_r)

    def get_link_position(self, q14):
        """Return the union of both arms' link positions. Used by the
        visualizer to draw both arms in one HTML/PNG."""
        ql, qr = self.split(q14)
        return (self.left.get_link_position(ql)
                + self.right.get_link_position(qr))

    # ----- IK helpers -----

    def inverse_kinematics(self, target_left, target_right, q_seed_14,
                           target_left_ori=None, target_right_ori=None,
                           orientation_weight=0.5,
                           **dls_kwargs):
        """Solve IK for both arms with multi-reference retry.

        target_left_ori / target_right_ori: optional 3x3 rotation matrices
        in the torso frame. Pass None on either side to do position-only
        IK on that arm. While the right wrist yaw motor is broken, leave
        target_right_ori=None and pass target_left_ori only.

        orientation_weight: scales the orientation error in the stacked
        task vector. Same weight is applied to both arms when their
        orientation tasks are active.

        Strategy: try several "natural reach" reference poses for the
        null-space bias. Return the first collision-free goal config.
        If all references produce in-collision configs, return the one
        with the lowest combined IK + collision-margin score.

        Each arm uses its own DLS solver independently. The seed (where
        the iteration starts) is always the current pose; the reference
        (where the null-space bias pulls toward) is what varies.
        """
        ql_seed, qr_seed = self.split(q_seed_14)

        # Mirror-symmetric reference pose options, ordered by likelihood
        # of working. Each entry is (left_ref_deg, right_ref_deg).
        # Use mirrored values (sign-flipped LSR/LSY/LWR/LWY) for the right.
        ref_poses_options = [
            # Default: forward reach at chest height
            ([0, 20, 0, 45, 0, 0, 0], [0, -20, 0, 45, 0, 0, 0]),
            # More elbow bend for closer/higher targets
            ([0, 25, 0, 70, 0, 0, 0], [0, -25, 0, 70, 0, 0, 0]),
            # More abduction for wider targets
            ([0, 40, 0, 45, 0, 0, 0], [0, -40, 0, 45, 0, 0, 0]),
            # More shoulder pitch for low targets (arm down and forward)
            ([-25, 20, 0, 45, 0, 0, 0], [-25, -20, 0, 45, 0, 0, 0]),
            # Straighter arm for far targets
            ([0, 15, 0, 20, 0, 0, 0], [0, -15, 0, 20, 0, 0, 0]),
        ]

        best_q = None
        best_score = float("inf")
        for k, (ref_l_deg, ref_r_deg) in enumerate(ref_poses_options):
            ref_l = np.radians(ref_l_deg)
            ref_r = np.radians(ref_r_deg)
            ql = self.left.inverse_kinematics(
                target_left, initial_position=ql_seed,
                target_orientation=target_left_ori,
                orientation_weight=orientation_weight,
                reference_pose=ref_l, **dls_kwargs)
            qr = self.right.inverse_kinematics(
                target_right, initial_position=qr_seed,
                target_orientation=target_right_ori,
                orientation_weight=orientation_weight,
                reference_pose=ref_r, **dls_kwargs)
            q = self.join(ql, qr)

            # Score: pure position error if in collision, else return
            # immediately. Score is used only as a fallback ranking.
            ee_l = self.left.get_end_effector_position(ql)
            ee_r = self.right.get_end_effector_position(qr)
            pos_err = (np.linalg.norm(ee_l - target_left)
                       + np.linalg.norm(ee_r - target_right))

            if not self.check_collision(q):
                # First collision-free is good enough -- return it.

                print(f"[IK] Used reference {k} for "
                        f"collision-free solution.")
                return q

            if pos_err < best_score:
                best_score = pos_err
                best_q = q

        # Nothing was collision-free. Return the closest-to-target one.
        # validate_goal in g1_plan_run.py will reject it with a clear
        # message, but at least the EE error will be informative.
        print(f"[IK] WARNING: no collision-free reference pose found. "
              f"Returning best-position attempt.")
        return best_q if best_q is not None else q


# ----- smoke test -----
if __name__ == "__main__":
    import math
    from myArm_v2 import _orientation_error, rpy_to_R

    dual = G1DualArm()
    print(f"Joints: {dual.num_joints}")
    print(f"Limits shape: {dual.joint_limits.shape}")

    # Both arms at clearance pose
    q_clear_l = np.radians([16, 18, 0, 58, 4, 1, 1])
    q_clear_r = np.radians([16, -18, 0, 58, 4, 1, 1])  # mirror
    q14 = dual.join(q_clear_l, q_clear_r)

    print(f"Within limits  : {dual.is_within_joint_limits(q14)}")
    print(f"In collision   : {dual.check_collision(q14)}  (expect False)")

    # --- Test 1: position-only IK (existing behavior unchanged) ---
    print("\nTest 1: position-only dual IK (both arms forward)")
    target_l = np.array([0.30,  0.10, 0.05])
    target_r = np.array([0.30, -0.10, 0.05])
    q_goal = dual.inverse_kinematics(target_l, target_r, q14)
    ee_l = dual.left.get_end_effector_position(q_goal[:7])
    ee_r = dual.right.get_end_effector_position(q_goal[7:])
    print(f"  pos err L: {np.linalg.norm(ee_l - target_l)*1000:.1f} mm")
    print(f"  pos err R: {np.linalg.norm(ee_r - target_r)*1000:.1f} mm")
    print(f"  collision : {dual.check_collision(q_goal)}  (expect False)")

    # --- Test 2: left-arm orientation, right position-only ---
    print("\nTest 2: left-arm orientation control, right position-only")
    # Palm pointing forward (+X torso) with thumb up (+Z torso).
    # Use rpy_to_R for a clean target: hand approach axis along +X,
    # rotated 0 about each torso axis = identity (hand X = torso X).
    R_left_target = rpy_to_R(0.0, 0.0, 0.0)  # identity
    q_goal = dual.inverse_kinematics(
        target_l, target_r, q14,
        target_left_ori=R_left_target,
        target_right_ori=None,
        orientation_weight=0.5)
    ee_l = dual.left.get_end_effector_position(q_goal[:7])
    ee_r = dual.right.get_end_effector_position(q_goal[7:])
    T_l = dual.left.get_end_effector_pose(q_goal[:7])
    ori_err = np.linalg.norm(_orientation_error(T_l[:3, :3], R_left_target))
    print(f"  pos err L  : {np.linalg.norm(ee_l - target_l)*1000:.1f} mm")
    print(f"  ori err L  : {np.degrees(ori_err):.1f} deg")
    print(f"  pos err R  : {np.linalg.norm(ee_r - target_r)*1000:.1f} mm")
    print(f"  collision  : {dual.check_collision(q_goal)}  (expect False)")
    print(f"  RWY angle R: {np.degrees(q_goal[13]):.2f} deg  "
          f"(unconstrained; will be whatever IK chose)")

    # --- Test 3: same as Test 2, but right RWY locked at current reading ---
    print("\nTest 3: left-orientation + right RWY locked at 0.300 rad")
    dual_locked = G1DualArm(locked_joints_right={6: 0.300})
    q_seed = q14.copy()
    q_seed[13] = 0.300  # match the lock
    q_goal = dual_locked.inverse_kinematics(
        target_l, target_r, q_seed,
        target_left_ori=R_left_target,
        target_right_ori=None,
        orientation_weight=0.5)
    ee_l = dual_locked.left.get_end_effector_position(q_goal[:7])
    ee_r = dual_locked.right.get_end_effector_position(q_goal[7:])
    T_l = dual_locked.left.get_end_effector_pose(q_goal[:7])
    ori_err = np.linalg.norm(_orientation_error(T_l[:3, :3], R_left_target))
    print(f"  pos err L  : {np.linalg.norm(ee_l - target_l)*1000:.1f} mm")
    print(f"  ori err L  : {np.degrees(ori_err):.1f} deg")
    print(f"  pos err R  : {np.linalg.norm(ee_r - target_r)*1000:.1f} mm")
    print(f"  RWY angle R: {np.degrees(q_goal[13]):.4f} deg  "
          f"(should be exactly 17.1885 deg = 0.300 rad)")
    print(f"  collision  : {dual_locked.check_collision(q_goal)}  "
          f"(expect False)")