#!/usr/bin/env python3
"""
g1_choreograph.py
=================
Run a sequence of pose+gesture steps on a G1Robot.

A "step" is a dict.

  Required:
    side  : "left" | "right" | "both"

  Target — specify exactly one form per step:

    Single-arm  (side="left" or "right"):
      target_xyz       : np.array(3)             Cartesian target (uses IK)
        target_R       :   3x3 rotation matrix    optional, with target_xyz
      target_q         : np.array(7)             direct joint-space target

    Dual-arm    (side="both"):
      target_xyz_left  : np.array(3)             both required for Cartesian
      target_xyz_right : np.array(3)
        target_R_left  :   3x3 rotation matrix    optional, with xyz pair
        target_R_right :   3x3 rotation matrix    optional, with xyz pair
      target_q         : np.array(14)            [left(7), right(7)]

    Relative joint goal (any side, exclusive with all the above):
      q_delta          : np.array(7) or (14)      q_goal = q_start + q_delta.
                         No IK, no Cartesian target. q_start is the previous
                         group's commanded endpoint (via chain_start), so this
                         is a pure relative joint move continuous with it —
                         e.g. lift = grasp pose with shoulder pitch -30°.

  Optional (any side):
    label           : string for logging
    skip_planning   : bool, default False — direct start→goal, no RRT.
                      Goal still gets a collision check. Only valid for
                      target_q steps.
    obstacles       : (N, 9) array-like, or [] for an empty world. When
                      present, IK / collision / RRT for THIS step run
                      against this obstacle set instead of the model's
                      baseline (the world the runner set on arm_model).
                      Transient: restored to baseline after the step, so a
                      step without the key falls back to baseline. This is
                      what lets a grasp express two worlds in one sequence —
                      box-present for the approach, box-dropped for the
                      grasp/lift — without separate run_sequence calls.
                      Steps with differing obstacle sets are never batched
                      together (each plans against its own world).
    gestures_before : {"left": name, "right": name} or None
    gestures_after  : {"left": name, "right": name} or None
    settle          : seconds to wait after execute (default 1.5)

The runner uses last-commanded q as the spline's start waypoint when it's
within 5° of measured (avoids the inter-trajectory bob); falls back to
measured otherwise for safety. Last-commanded is tracked per arm and
reconstructed for "both" steps from the left/right history.

Sequences may mix sides freely. If any step is "both", the caller must
pass a G1DualArm as arm_model. Arm-model side dispatch is automatic.
"""

import math
import time
import threading
import numpy as np

from rrt_planner import RRTConnect


# Hand slew is ~60 deg/s; ~90° gesture = 1.5 s. 1.0 s wait covers small
# gestures; called once per before/after hook.
_GESTURE_WAIT_S = 1.0

# Per-joint safety threshold for using last-commanded q as the next
# trajectory's start waypoint (the chain_start optimization). Under this
# threshold, the spline starts at the commanded endpoint of the previous
# trajectory → no PD discontinuity between segments, no sag bob. Over it,
# something has gone wrong (motor stalled, external push, prior abort)
# and we silently fall back to measured to avoid yanking the arm.
_MAX_DRIFT_RAD = math.radians(5.0)


# ---------------------------------------------------------------------------
# Per-step obstacle override
# ---------------------------------------------------------------------------
# A step may carry its own `obstacles`. We swap it onto the model (and, for a
# dual model, its two sub-arms — IK/RRT for per-side steps run through the
# sub-arm, and G1DualArm.check_collision also reads the sub-arms) just for the
# duration of that step's resolve+plan, then restore. Transient by design:
# the baseline the runner set on arm_model is what a keyless step sees.

def _normalize_obstacles(obs):
    """Return an (N, 9) float array from a step's `obstacles` value, or None
    if the step doesn't override (meaning: use the model's current world).
    An explicit empty value ([] / empty array) means 'no obstacles'."""
    if obs is None:
        return None
    arr = np.asarray(obs, dtype=float)
    if arr.size == 0:
        return np.empty((0, 9))
    if arr.ndim != 2 or arr.shape[1] != 9:
        raise ValueError(
            f"step 'obstacles' must be (N, 9) or empty; got shape {arr.shape}")
    return arr


def _set_obstacles(arm_model, obs_arr):
    """Swap `obs_arr` onto arm_model (and sub-arms if dual). Returns a restore
    token for _restore_obstacles. No-op (returns None) when obs_arr is None."""
    if obs_arr is None:
        return None
    if hasattr(arm_model, "left") and hasattr(arm_model, "right"):
        token = (arm_model.obstacles,
                 arm_model.left.obstacles,
                 arm_model.right.obstacles)
        arm_model.obstacles       = obs_arr
        arm_model.left.obstacles  = obs_arr
        arm_model.right.obstacles = obs_arr
    else:
        token = (arm_model.obstacles,)
        arm_model.obstacles = obs_arr
    return token


def _restore_obstacles(arm_model, token):
    """Undo _set_obstacles. Safe to call with a None token."""
    if token is None:
        return
    if hasattr(arm_model, "left") and hasattr(arm_model, "right"):
        (arm_model.obstacles,
         arm_model.left.obstacles,
         arm_model.right.obstacles) = token
    else:
        (arm_model.obstacles,) = token


def _same_obstacles(step_a, step_b):
    """True if two steps resolve to the same obstacle world (for batching).
    Two keyless steps match (both fall back to baseline). A keyless step and
    an explicit-world step never match — they must not share a batch."""
    oa = _normalize_obstacles(step_a.get("obstacles"))
    ob = _normalize_obstacles(step_b.get("obstacles"))
    if oa is None and ob is None:
        return True
    if (oa is None) != (ob is None):
        return False
    return oa.shape == ob.shape and np.array_equal(oa, ob)


def preflight_validate(arm_model, steps, verbose=True, obstacle_names=None):
    """Validate every step's goal before any motion happens.

    obstacle_names : optional, for nicer collision diagnostics. Either a list
        parallel to the obstacle world (e.g. ["torso","box","table"]) applied
        to every step, or a callable step -> list (use this when steps span
        different worlds — e.g. box-present vs box-dropped — so each gets the
        right names). When omitted, a colliding goal still reports the pairs
        but labels obstacles by index ("obstacle 1").

    Walks the sequence using *commanded* q chaining: each step's q_goal
    becomes the IK seed for the next step on the same side. The very
    first step per side seeds IK with zeros — a defensive default that
    only matters if the first step on a side uses target_pose (most
    choreographies start with a joint-space clearance pose, so this is
    rarely the binding seed).

    Per step, checks:
      1. Goal resolves — target_q is well-formed, or IK on target_pose
         returns a usable solution.
      2. q_goal is within URDF joint limits.
      3. q_goal is collision-free against arm_model.obstacles.

    Important caveat — locked joints:
      If arm_model.locked_joints (or .left/.right.locked_joints) is empty,
      validation runs against the full URDF range. At runtime, locked-joint
      substitution replaces those indices with their hardware-pinned angle
      before the same checks run. This means preflight is *more permissive*
      than runtime for locked joints: a preflight pass guarantees the
      unlocked joints are valid, and locked joints will be substituted to
      their hardware values which are themselves in URDF range. Preflight
      failures are definite failures; preflight successes are 'very likely
      OK' with runtime handling the locked-joint particulars.

    Returns (ok, issues) where issues is a list of (step_idx, level, msg).
    Level is 'ERROR' (blocks confirmation) or 'WARN' (informational only).
    """
    issues = []
    chain_starts = {"left": None, "right": None, "both": None}

    for i, step in enumerate(steps):
        side = step.get("side")
        if side not in ("left", "right", "both"):
            issues.append((i, "ERROR",
                f"side must be 'left', 'right', or 'both'; got {side!r}"))
            continue

        sub = _arm_for_step(arm_model, side)
        n_dof = 14 if side == "both" else 7

        # Per-step obstacle world. Without this, a grasp step (planned in a
        # box-dropped world) would be collision-checked here against the
        # box-present baseline and falsely flagged. Normalize first so a
        # malformed value is a reportable ERROR, not a crash.
        try:
            obs_arr = _normalize_obstacles(step.get("obstacles"))
        except Exception as e:
            issues.append((i, "ERROR", f"bad 'obstacles': {e}"))
            continue
        token = _set_obstacles(arm_model, obs_arr)
        try:
            # IK seed: chain from previous step on the same side; for 'both'
            # also accept a synthesis from prior single-side commanded q's;
            # else zeros.
            if chain_starts[side] is not None:
                q_seed = chain_starts[side]
            elif (side == "both"
                  and chain_starts["left"] is not None
                  and chain_starts["right"] is not None):
                q_seed = np.concatenate([chain_starts["left"],
                                         chain_starts["right"]])
            else:
                q_seed = np.zeros(n_dof)

            # Resolve goal (target_q direct, target_pose → IK)
            try:
                q_goal = _resolve_goal(arm_model, step, q_seed)
            except Exception as e:
                issues.append((i, "ERROR", f"goal resolution failed: {e}"))
                continue

            if q_goal is None:
                issues.append((i, "ERROR",
                    "IK returned no solution (target_pose unreachable from seed)"))
                continue

            q_goal = np.asarray(q_goal, dtype=float)
            if q_goal.shape != (n_dof,):
                issues.append((i, "ERROR",
                    f"goal has wrong shape: expected ({n_dof},), "
                    f"got {tuple(q_goal.shape)}"))
                continue

            # Joint limits — report ALL violators so the user sees the full picture
            tol = getattr(type(sub), "JOINT_LIMIT_TOL", 1e-9)
            for j in range(len(q_goal)):
                lo = sub.joint_limits[j, 0]
                hi = sub.joint_limits[j, 1]
                if q_goal[j] < lo - tol:
                    issues.append((i, "ERROR",
                        f"joint {j} = {math.degrees(q_goal[j]):+.1f}° below "
                        f"URDF limit [{math.degrees(lo):+.1f}°, "
                        f"{math.degrees(hi):+.1f}°]"))
                elif q_goal[j] > hi + tol:
                    issues.append((i, "ERROR",
                        f"joint {j} = {math.degrees(q_goal[j]):+.1f}° above "
                        f"URDF limit [{math.degrees(lo):+.1f}°, "
                        f"{math.degrees(hi):+.1f}°]"))

            # Collision (against this step's world)
            try:
                if sub.check_collision(q_goal):
                    names = (obstacle_names(step) if callable(obstacle_names)
                             else obstacle_names)
                    pairs = []
                    if hasattr(sub, "collision_report"):
                        pairs = sub.collision_report(q_goal, names)
                    msg = "goal pose collides with obstacle(s)"
                    if pairs:
                        msg += ": " + "; ".join(f"{cap} vs {obs}"
                                                for cap, obs in pairs)
                    issues.append((i, "ERROR", msg))
            except Exception as e:
                issues.append((i, "WARN", f"collision check raised: {e}"))

            # Chain forward for the next step on this side
            if side == "left":
                chain_starts["left"] = q_goal.copy()
            elif side == "right":
                chain_starts["right"] = q_goal.copy()
            else:  # both
                chain_starts["left"] = q_goal[:7].copy()
                chain_starts["right"] = q_goal[7:].copy()
                chain_starts["both"] = q_goal.copy()
        finally:
            _restore_obstacles(arm_model, token)

    errors = [it for it in issues if it[1] == "ERROR"]
    warnings = [it for it in issues if it[1] == "WARN"]
    ok = len(errors) == 0

    if verbose:
        print(f"\n[preflight] Validating {len(steps)} steps...")
        if not issues:
            print(f"[preflight] OK — all {len(steps)} goals reachable "
                  f"and collision-free.")
        else:
            tag = "FAILED" if errors else "OK (warnings only)"
            print(f"[preflight] {tag}: {len(errors)} error(s), "
                  f"{len(warnings)} warning(s):")
            for idx, lvl, msg in issues:
                step_label = steps[idx].get("label", f"step {idx+1}")
                print(f"  [{lvl}] step {idx+1} '{step_label}': {msg}")

    return ok, issues


def run_sequence(robot, arm_model, steps, confirm=None, record=None):
    """Execute a list of choreography steps, batching consecutive same-side
    steps with no gestures-between into single streamed trajectories so the
    arm doesn't decelerate to zero between every step.

    Threads the last-commanded q forward per side so each group's spline
    starts from the previous group's commanded endpoint (when safe) rather
    than the sagged measured position — eliminates the inter-trajectory
    bob caused by PD steady-state error during settles.

    `arm_model` may be a single G1Arm (for left- or right-only sequences)
    or a G1DualArm (required if any step has side='both'; also fine for
    pure left/right sequences). For per-side steps under a dual model,
    we dispatch IK / limits / collision checks to arm_model.left or .right.

    RRT step size and iteration cap are not arguments — they live as
    defaults on RRTConnect. Tune them there if needed.

    Obstacle worlds: arm_model.obstacles is the baseline for the whole
    sequence. A step may override it for itself via an `obstacles` key
    (e.g. box-present for the approach, box-dropped for the grasp/lift);
    the override is transient and steps with different worlds are never
    batched together.

    confirm: optional callable. Called once before each group executes, as
    confirm(group_index, num_groups, group_steps) -> bool. Return False to
    stop the sequence before that group runs (the arm holds its current
    commanded pose; nothing is executed for that group or any after it).
    Return True to proceed. When None (default), every group runs without
    prompting — behaviour is unchanged for callers that don't pass it.
    Running the whole sequence in one run_sequence call with a confirm
    callback gives per-phase gating AND threads last_commanded across the
    whole grab, so each phase's IK seeds from the previous phase's commanded
    endpoint instead of re-reading the settled (sagged) measured pose.

    record: optional list. When given, each executed group appends a dict
    {label, side, obstacles, path, planned} to it — the exact waypoints the
    runner planned/streamed for that group, the world they were planned in,
    and whether RRT ran (planned=True) or it was a direct skip-planning move.
    Lets a caller render an HTML of what actually ran, with no extra planning.
    """
    is_dual = (hasattr(arm_model, "left") and hasattr(arm_model, "right"))

    # Pre-flight: if any "both" step, we need a dual model.
    if any(s.get("side") == "both" for s in steps) and not is_dual:
        raise ValueError("Sequence contains side='both' steps but arm_model "
                         "is not a G1DualArm")

    # Sync obstacle list to subarms. G1DualArm.check_collision does this on
    # the fly, but per-side steps go through arm_model.left/.right directly,
    # so without this sync those subarms would see an empty obstacle list.
    if is_dual:
        arm_model.left.obstacles = arm_model.obstacles
        arm_model.right.obstacles = arm_model.obstacles

    last_commanded = {"left": None, "right": None}
    groups = _group_for_streaming(steps)
    for gi, group in enumerate(groups):
        side = group[0]["side"]

        # Per-group confirm gate. Checked BEFORE any gesture/motion for the
        # group, so declining leaves the arm exactly where it is.
        if confirm is not None and not confirm(gi, len(groups), group):
            print(f"\n[choreograph] stopped before group {gi+1}/{len(groups)} "
                  f"by confirm callback — arm holds current pose.")
            return

        # Build chain_start in the right shape for the upcoming group.
        if side == "both":
            if (last_commanded["left"] is not None
                    and last_commanded["right"] is not None):
                chain_start = np.concatenate(
                    [last_commanded["left"], last_commanded["right"]])
            else:
                chain_start = None
        else:
            chain_start = last_commanded[side]

        if len(group) == 1:
            print(f"\n--- step {gi+1}/{len(groups)}: "
                  f"{group[0].get('label', '?')} ---")
            q_end = _run_step(robot, arm_model, group[0],
                              chain_start=chain_start, record=record)
        else:
            labels = " → ".join(s.get("label", "?") for s in group)
            print(f"\n--- batch {gi+1}/{len(groups)} ({len(group)} steps): "
                  f"{labels} ---")
            q_end = _run_batch(robot, arm_model, group,
                               chain_start=chain_start, record=record)

        # Update last_commanded so the next group sees this endpoint.
        if side == "both":
            last_commanded["left"]  = q_end[:7]
            last_commanded["right"] = q_end[7:]
        else:
            last_commanded[side] = q_end
    print("\n[choreograph] sequence complete.")


def _arm_for_step(arm_model, side):
    """Return the sub-model to use for IK / limits / collision checks.
    With a G1DualArm parent, dispatches side='left'/'right' to its
    sub-arms and side='both' to the parent. With a single G1Arm parent,
    returns it for left/right and raises for 'both'."""
    if hasattr(arm_model, "left") and hasattr(arm_model, "right"):
        if side == "left":
            return arm_model.left
        if side == "right":
            return arm_model.right
        return arm_model
    if side == "both":
        raise ValueError("side='both' step requires arm_model to be a "
                         "G1DualArm; got a single-arm model")
    return arm_model


def _safe_chain_start(robot, side, override):
    """Return what to feed as the new trajectory's first waypoint.

    If `override` (last-commanded q) is within _MAX_DRIFT_RAD of measured
    on every joint, return it — the new spline starts where the previous
    one was driving toward, target_q is continuous across the boundary,
    no bob. Otherwise return measured: the commanded position has drifted
    too far to trust as a starting point, and a spline starting there
    would yank the arm. Accepts the slight bob as the safer option."""
    measured = robot.arm.get_current_q(side)
    if override is None:
        return measured
    drift = float(np.max(np.abs(np.asarray(override) - measured)))
    if drift > _MAX_DRIFT_RAD:
        print(f"  [info] last-commanded q diverged from measured by "
              f"{math.degrees(drift):.1f}° (> {math.degrees(_MAX_DRIFT_RAD):.0f}°); "
              f"using measured")
        return measured
    return override


def _group_for_streaming(steps):
    """Group consecutive steps that can stream as one trajectory:
       - same side
       - current step has no gestures_after
       - next step has no gestures_before
       - same skip_planning flag
       - same obstacle world (each plan needs one world; a box-present and a
         box-dropped step must not share a batch)
    Cuts are made wherever any of these differ."""
    groups, cur = [], []
    for s in steps:
        if not cur:
            cur.append(s); continue
        prev = cur[-1]
        cut = (
            s["side"] != prev["side"]
            or prev.get("gestures_after")
            or s.get("gestures_before")
            or bool(s.get("skip_planning")) != bool(prev.get("skip_planning"))
            or not _same_obstacles(prev, s)
        )
        if cut:
            groups.append(cur); cur = [s]
        else:
            cur.append(s)
    if cur:
        groups.append(cur)
    return groups


def _run_batch(robot, arm_model, group, chain_start=None, record=None):
    """Resolve each step in the group, concatenate the waypoints, and call
    execute_trajectory once. Hand gestures only fire on the first step's
    gestures_before and the last step's gestures_after.

    If chain_start is given and within drift threshold of measured, the
    concatenated path starts from chain_start instead of measured — keeps
    target_q continuous with the previous group's commanded endpoint.

    Returns the final commanded q for this group (= last step's q_goal),
    which the caller threads forward as the next group's chain_start.

    All steps in a group share one obstacle world — _group_for_streaming
    cuts the batch whenever `obstacles` differs — so we set it once from
    group[0] and restore after planning.
    """
    side = group[0]["side"]
    sub = _arm_for_step(arm_model, side)

    _apply_gestures(robot, group[0].get("gestures_before"))

    full_path = []
    obs_token = _set_obstacles(
        arm_model, _normalize_obstacles(group[0].get("obstacles")))
    world = np.array(sub.obstacles, dtype=float)
    try:
        q_chain = _snap_locked(sub, _safe_chain_start(robot, side, chain_start))
        for step in group:
            q_goal = _resolve_goal(arm_model, step, q_chain)
            if not sub.is_within_joint_limits(q_goal):
                raise RuntimeError(f"[{step.get('label')}] goal outside joint limits")
            if sub.check_collision(q_goal):
                raise RuntimeError(f"[{step.get('label')}] goal in collision")

            if step.get("skip_planning"):
                seg = [q_chain, q_goal]
            else:
                seg = _plan_rrt(sub, q_chain, q_goal)
                if not seg:
                    raise RuntimeError(f"[{step.get('label')}] planner found no path")

            # Drop leading waypoint on every segment except the first to avoid
            # duplicate points where seg ends meet.
            full_path.extend(seg if not full_path else seg[1:])
            q_chain = q_goal
    finally:
        _restore_obstacles(arm_model, obs_token)

    if record is not None:
        record.append({
            "label":     " → ".join(s.get("label", "?") for s in group),
            "side":      side,
            "obstacles": world,
            "path":      [np.asarray(q, dtype=float).copy() for q in full_path],
            "planned":   any(not s.get("skip_planning") for s in group),
        })

    print(f"  concatenated {len(full_path)} waypoints across "
          f"{len(group)} steps")
    robot.arm.execute_trajectory(full_path, side=side, dwell=0.0)

    _apply_gestures(robot, group[-1].get("gestures_after"))
    settle = float(group[-1].get("settle", 1.5))
    if settle > 0:
        time.sleep(settle)
    return q_chain


def _apply_locked_joints(arm_model, side, q_goal, label=""):
    """If arm_model has locked joints, overwrite the matching indices of
    q_goal with the locked values. Works for single G1Arm and G1DualArm.

    The user's target_q value for any locked joint is meaningless (the
    motor can't reach a non-locked position) so we silently replace it
    with the locked angle. This keeps user-facing target_q constants
    (e.g. _REST_RIGHT) portable across runs where the locked angle
    differs — once the motor is fixed and the lock is removed, the
    helper becomes a no-op and the user's value is honored again.

    Emits one log line per index where the substitution actually changed
    a value (FP-tolerant), so users notice when their hand-tuned target
    is being overridden. Quiet in the common case where the user already
    set the locked angle.
    """
    _LOG_TOL_RAD = 1e-6   # below this, treat as "already locked" — no log

    def _maybe_log(idx_in_q, locked_ang):
        old = float(q_goal[idx_in_q])
        if abs(old - locked_ang) > _LOG_TOL_RAD:
            tag = f" '{label}'" if label else ""
            print(f"  [locked] step{tag} joint {idx_in_q} "
                  f"target {math.degrees(old):+.1f}° → "
                  f"locked {math.degrees(locked_ang):+.1f}°")

    if hasattr(arm_model, "left") and hasattr(arm_model, "right"):
        # Dual model: locked_joints live on the left/right sub-arms.
        if side in ("left", "both"):
            for j, ang in getattr(arm_model.left, "locked_joints", {}).items():
                _maybe_log(j, ang)
                q_goal[j] = ang
        if side in ("right", "both"):
            offset = 7 if side == "both" else 0
            for j, ang in getattr(arm_model.right, "locked_joints", {}).items():
                _maybe_log(offset + j, ang)
                q_goal[offset + j] = ang
    else:
        for j, ang in getattr(arm_model, "locked_joints", {}).items():
            _maybe_log(j, ang)
            q_goal[j] = ang
    return q_goal


def _resolve_goal(arm_model, step, q_start):
    """Return q_goal for a step. target_q wins if present; otherwise IK
    from the appropriate Cartesian target(s), seeded by q_start.

    Dispatches by step['side']:
      'left'/'right' → single-arm IK on arm_model.left/.right (or the
                       given single arm), 7-vec target_q or target_xyz
                       + optional target_R.
      'both'         → dual IK on arm_model itself, 14-vec target_q or
                       target_xyz_left+target_xyz_right with optional
                       target_R_left/right.
    """
    side = step["side"]

    if "q_delta" in step:
        # Relative joint goal: q_goal = q_start + q_delta. No IK. Used e.g. by
        # lift, which is "the grasp pose, shoulders pitched up" — q_start is
        # the previous group's commanded endpoint (the grasp pose) threaded in
        # via chain_start, so this needs no Cartesian target and no IK solve,
        # keeping the grasp->lift transition continuous in joint space.
        dq = np.asarray(step["q_delta"], dtype=float)
        expected = 14 if side == "both" else 7
        if dq.shape != (expected,):
            raise ValueError(
                f"q_delta must be length {expected} for side='{side}', "
                f"got shape {dq.shape}")
        q_goal = np.asarray(q_start, dtype=float) + dq
        return _apply_locked_joints(arm_model, side, q_goal,
                                    label=step.get("label", ""))

    if "target_q" in step:
        q_goal = np.asarray(step["target_q"], dtype=float)
        expected = 14 if side == "both" else 7
        if q_goal.shape != (expected,):
            raise ValueError(
                f"target_q must be length {expected} for side='{side}', "
                f"got shape {q_goal.shape}")
        return _apply_locked_joints(arm_model, side, q_goal,
                                    label=step.get("label", ""))

    sub = _arm_for_step(arm_model, side)

    if side == "both":
        if "target_xyz_left" not in step or "target_xyz_right" not in step:
            raise ValueError("side='both' Cartesian step needs both "
                             "target_xyz_left and target_xyz_right")
        tl = np.asarray(step["target_xyz_left"],  dtype=float)
        tr = np.asarray(step["target_xyz_right"], dtype=float)
        Rl = step.get("target_R_left")
        Rr = step.get("target_R_right")
        return sub.inverse_kinematics(
            tl, tr, q_start,
            target_left_ori=Rl, target_right_ori=Rr)

    # single arm
    target = np.asarray(step["target_xyz"], dtype=float)
    R = step.get("target_R")
    if R is not None:
        return sub.inverse_kinematics(
            target, initial_position=q_start,
            target_orientation=R)
    return sub.inverse_kinematics(target, initial_position=q_start)


def _run_step(robot, arm_model, step, chain_start=None, record=None):
    """Run a single (un-batched) step.

    Same chain_start contract as _run_batch: if provided and within drift
    threshold, use it as the spline's first waypoint and the IK seed,
    otherwise fall back to measured. Returns the step's q_goal so the
    caller can thread it forward as the next group's chain_start.
    """
    side = step.get("side")
    if side not in ("left", "right", "both"):
        raise ValueError(f"step needs side in {{left, right, both}}, "
                         f"got {side!r}")
    if robot.arm is None:
        raise RuntimeError("G1Robot constructed without arm — can't run steps")

    # Validate target form against side.
    has_xyz       = "target_xyz" in step
    has_xyz_dual  = ("target_xyz_left" in step
                     or "target_xyz_right" in step)
    has_q         = "target_q" in step
    has_dq        = "q_delta" in step

    # q_delta is a relative joint goal (q_start + q_delta, no IK). It's
    # mutually exclusive with any absolute target and resolved in _resolve_goal.
    if has_dq and (has_q or has_xyz or has_xyz_dual):
        raise ValueError("q_delta is exclusive with target_q / target_xyz*")
    if has_dq:
        pass  # form is valid; _resolve_goal handles shape per side
    elif side == "both":
        if has_xyz:
            raise ValueError("side='both' uses target_xyz_left+_right, "
                             "not target_xyz")
        # need exactly one of: target_q, or BOTH target_xyz_left + _right
        if has_q and has_xyz_dual:
            raise ValueError("side='both' step takes target_q OR "
                             "target_xyz_left+target_xyz_right, not both")
        if not has_q and not has_xyz_dual:
            raise ValueError("side='both' step needs target_q or "
                             "target_xyz_left+target_xyz_right")
    else:
        if has_xyz_dual:
            raise ValueError(f"side='{side}' uses target_xyz, "
                             f"not target_xyz_left/_right")
        if has_xyz == has_q:
            raise ValueError("step needs exactly one of target_xyz / target_q")

    # ---- before-gestures ----
    _apply_gestures(robot, step.get("gestures_before"))

    # ---- resolve target ----
    # q_start serves three purposes: IK seed, RRT plan start, and
    # path[0] for skip_planning. Use chain_start when safe so all three
    # are consistent with the previous group's commanded endpoint.
    #
    # IK / collision / RRT below run against this step's obstacle world if it
    # declares one; restored to baseline before execute (execution doesn't
    # read obstacles, but we restore promptly so a later keyless step or an
    # exception path sees the baseline).
    obs_token = _set_obstacles(arm_model, _normalize_obstacles(step.get("obstacles")))
    try:
        sub = _arm_for_step(arm_model, side)
        world = np.array(sub.obstacles, dtype=float)
        q_start = _snap_locked(sub, _safe_chain_start(robot, side, chain_start))
        q_goal  = _resolve_goal(arm_model, step, q_start)

        # ---- safety: goal must be in limits and collision-free ----
        if not sub.is_within_joint_limits(q_goal):
            raise RuntimeError(f"[{step.get('label', '')}] goal outside joint limits")
        if sub.check_collision(q_goal):
            raise RuntimeError(f"[{step.get('label', '')}] goal in collision")

        # ---- plan ----
        skip_planning = step.get("skip_planning", False)
        has_any_xyz = has_xyz or has_xyz_dual
        if skip_planning and has_any_xyz:
            print("  [warn] skip_planning=True with Cartesian target; "
                  "using direct start→goal anyway")
        if skip_planning:
            path = [q_start, q_goal]
            print(f"  direct path (skip_planning): "
                  f"max joint Δ = {np.degrees(np.max(np.abs(q_goal - q_start))):.1f}°")
        else:
            path = _plan_rrt(sub, q_start, q_goal)
            if not path:
                raise RuntimeError(f"[{step.get('label', '')}] planner found no path")
            print(f"  planned path: {len(path)} waypoints")
    finally:
        _restore_obstacles(arm_model, obs_token)

    if record is not None:
        record.append({
            "label":     step.get("label", "?"),
            "side":      side,
            "obstacles": world,
            "path":      [np.asarray(q, dtype=float).copy() for q in path],
            "planned":   not skip_planning,
        })

    # ---- execute ----
    robot.arm.execute_trajectory(path, side=side, dwell=0.0)

    # ---- after-gestures ----
    _apply_gestures(robot, step.get("gestures_after"))

    # ---- settle ----
    settle = float(step.get("settle", 1.5))
    if settle > 0:
        time.sleep(settle)
    return q_goal


def _snap_locked(arm_model, q):
    """Force locked joints in q to exactly their locked angle.

    Locking clamps a joint's limits to a zero-width interval [lock, lock].
    A q from elsewhere (measured readback, chain_start, IK) carries the
    locked joint's value with its own floating-point tail, which generally
    differs from `lock` by more than JOINT_LIMIT_TOL (1e-9) — so
    is_within_joint_limits, and thus the RRT start/goal check, rejects it
    even though it prints as equal. Snapping makes the value bit-equal to the
    limit, landing the point exactly on the zero-width interval. Same
    substitution _resolve_goal does for target_q goals; applied here to the
    RRT endpoints too.
    """
    q = np.asarray(q, dtype=float).copy()
    if hasattr(arm_model, "left") and hasattr(arm_model, "right"):
        n_left = arm_model.left.num_joints  # right active idx j -> global j+n_left
        for j, ang in arm_model.left.locked_joints.items():
            q[j] = ang
        for j, ang in arm_model.right.locked_joints.items():
            q[n_left + j] = ang
    else:
        for j, ang in getattr(arm_model, "locked_joints", {}).items():
            q[j] = ang
    return q


def _plan_rrt(arm_model, q_start, q_goal):
    """RRT-Connect plan with shortcut polish — mirrors g1_plan_run.plan().

    Snaps locked joints onto their zero-width limits before planning: a q
    from measured readback / chain_start / IK carries the locked joint's
    value with a floating-point tail that generally differs from the locked
    angle by more than JOINT_LIMIT_TOL, so the RRT start/goal would read as
    out-of-limits and the planner would never expand past step 0. See
    _snap_locked.
    """
    from g1_plan_run import plan as _plan
    q_start = _snap_locked(arm_model, q_start)
    q_goal  = _snap_locked(arm_model, q_goal)
    return _plan(arm_model, q_start, q_goal)


def _apply_gestures(robot, gestures):
    """Fire gestures on whichever hands are present.
    `gestures` is {"left": name, "right": name} or None."""
    if not gestures:
        return
    fired = False
    for side, name in gestures.items():
        if name is None:
            continue
        ctrl = robot.left_hand if side == "left" else robot.right_hand
        if ctrl is None:
            print(f"  [warn] gesture '{name}' requested for {side} hand "
                  f"but G1Robot was built without it; skipping")
            continue
        ctrl.set_gesture(name)
        fired = True
    if fired:
        time.sleep(_GESTURE_WAIT_S)