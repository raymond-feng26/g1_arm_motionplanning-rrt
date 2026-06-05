#!/usr/bin/env python3
"""
choreographies.py
=================
Runtime sequence builders for the grasp task, consumed by g1_choreograph.run_sequence.

Each builder returns a list of step dicts (see g1_choreograph.py for the step
schema). These are built at runtime — not static — because the targets and
obstacle worlds come from the live AprilTag scene:

  build_grab(...)      approach -> close in -> grip -> lift     (grab_run.py)
  build_put_down(...)  place -> push right -> retreat to rest   (grab_run.py)
  build_pull_left(...) right hand to a box jammed on the right  (pull_left_run.py)

Shared poses (_REST / _REST_RIGHT / _LIFT_LEFT / _LIFT_RIGHT) live at the top.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Joint indices for the left arm (and right arm — same convention):
#   0 shoulder pitch   3 elbow            6 wrist yaw
#   1 shoulder roll    4 wrist roll
#   2 shoulder yaw     5 wrist pitch
# ---------------------------------------------------------------------------


# ===========================================================================
# Shared poses — clearance/rest + lift, used by build_grab and build_put_down.
# (Tune visually on the robot; angles are in degrees, converted to radians.)
# ===========================================================================

_REST = np.radians([
     +15.53,   # shoulder pitch
     +10.09,   # shoulder roll
      -1.64,   # shoulder yaw
     +57.17,   # elbow
      +4.60,   # wrist roll
      +1.02,   # wrist pitch
      +1.93,   # wrist yaw
])

# Mirror clearance pose for the right arm: shoulder roll = -20° (out to the
# right side). Symmetric to _REST; used as the right arm's idle pose during
# left-only motions in dual-arm sequences.
_REST_RIGHT = np.radians([14.16, -10.78, 0.0, 58.26, -9.01, 3.31, 3.61])
_LIFT_RIGHT = np.radians([
     -46.72,   # shoulder pitch
      -9.54,   # shoulder roll
     -15.95,   # shoulder yaw
     +34.68,   # elbow
      +1.93,   # wrist roll
     -18.13,   # wrist pitch
      -4.72,   # wrist yaw
])
_LIFT_LEFT= np.radians([
     -53.97,   # shoulder pitch
      +7.49,   # shoulder roll
      +8.52,   # shoulder yaw
     +48.25,   # elbow
      -4.53,   # wrist roll
     -25.93,   # wrist pitch
      -3.54,   # wrist yaw
])


# ===========================================================================
# GRAB + LIFT  (runtime builder — NOT a static CHOREOGRAPHIES entry)
# ---------------------------------------------------------------------------
# Unlike the choreographies above, a grab can't be assembled at import time:
# the targets and the two obstacle worlds come from the live AprilTag at
# runtime. grab_run.py snapshots the scene, then calls build_grab(...) and
# feeds the result straight into g1_choreograph.run_sequence.
#
# Two obstacle worlds (Phase-1 per-step `obstacles` key):
#   obs_full   — box present; the approach plans AROUND it.
#   obs_no_box — box dropped; the grasp/lift intentionally contact it.
# Both already include _default_obstacles() + the table slab; the builder
# never sees the camera, only the finished targets and worlds.
#
# Gesture timing — the asymmetry is deliberate:
#   CLOSE  rides approach.gestures_before. Closing can be lazy: nothing is
#          near the hands while sweeping in to pre-grasp, and the hand's own
#          100 Hz loop keeps slewing shut even after the arm starts moving.
#   OPEN   rides approach.gestures_AFTER, i.e. it fires once the arm reaches
#          pre-grasp — NOT at the start of close-in. Opening from a closed
#          fist is a big finger sweep, and the fingers must be fully open
#          before they slot around the box faces. Hanging it on approach's
#          tail lets the pre-grasp `settle` absorb the travel.
#   GRAB   rides grasp.gestures_after — bend the index into the slot once the
#          hands are seated on the faces.
#   LIFT   no gesture; the hand loop holds the grab pose on its own thread.
#
# Orientation (tilted boxes): the left arm tracks R_left = Rz(box yaw) so the
# left hand stays square to the yawed box; the right arm keeps R_right =
# identity (square to torso, as originally validated) — its wrist-yaw motor is
# locked and set by hand, so this is a soft bias the IK can't fully meet. The
# choreograph IK path uses G1DualArm.inverse_kinematics's default
# orientation_weight (0.5). Pass R_left/R_right into build_grab.
#
# Note on the right hand: only the right *wrist-yaw arm joint* (RWY/13) is
# locked. The right dex3 hand is a separate subsystem and gestures normally,
# so both hands get close/open/grab.
# ===========================================================================

def build_grab(targets, obs_full, obs_no_box, *,
               clearance=True,
               clearance_settle=0.1,
               lift_sp_deg=-30.0,
               approach_settle=0.3,
               grasp_settle=0.2,
               lift_settle=0.1,
               hands=("left", "right"),
               R_left=None, R_right=None):
    """Return the grab+lift step list from runtime perception outputs.

    Parameters
    ----------
    targets         : dict from grasp_target.grasp_targets(); requires keys
                      left_pre, right_pre, left_grasp, right_grasp
                      (base-frame xyz np arrays).
    obs_full        : (N, 9) world WITH the box  — for the approach leg.
    obs_no_box      : (N, 9) world WITHOUT the box — for grasp + lift.
    clearance       : True  -> prepend a move to the _REST/_REST_RIGHT
                              clearance pose (skip_planning, joint-space) so
                              the approach starts from a known, repeatable
                              configuration instead of wherever the arm was;
                      False -> no clearance step;
                      a 14-vec -> use that as the clearance target_q.
                      NOTE: skip_planning=True only collision-checks the
                      clearance GOAL, not the swept path. _REST is a tucked
                      pose, so retracting to it from a benign start is clear;
                      if the arm may start extended over the box, move it to
                      a safe pose first or set clearance=False.
    clearance_settle: dwell after reaching the clearance pose.
    lift_sp_deg     : degrees to add to BOTH shoulder-pitch joints (index 0
                      of each arm) at the lift, relative to the grasp pose.
                      Negative pitches the shoulders up (raises the box).
                      Joint-space, so the lift needs no IK and stays
                      continuous with the grasp — see step 3.
    approach_settle : dwell at pre-grasp. Sized to let OPEN fully complete
                      before close-in — keep this >= ~2 s for an open-fist
                      sweep.
    grasp_settle    : dwell after reaching grasp, lets GRAB seat the grip
                      before the lift pulls on the box.
    lift_settle     : dwell after the lift.
    hands           : which hands to drive gestures on.
    R_left, R_right : optional 3x3 approach-orientation targets (torso frame)
                      applied to the approach + close-in steps. The grab passes
                      R_left = Rz(box yaw) (left hand square to the yawed box)
                      and R_right = identity (right square to torso, as
                      originally validated; its wrist-yaw is locked). None on
                      either arm = position-only IK for that arm.

    Gesture placement: CLOSE rides the clearance step's gestures_before when
    a clearance step exists (so the hands close earliest, with the clearance
    + approach motion to finish); otherwise it falls back onto the approach
    step. OPEN/GRAB are unaffected (approach.gestures_after / grasp.gestures_after).

    The grasp step's IK is seeded (via run_sequence's chain_start) from the
    approach step's commanded q — same homotopy-continuity trick used
    manually in grasp_plan_hardware (q_grasp seeded from q_pregrasp).
    """
    def _g(name):
        return {s: name for s in hands}

    left_pre,  right_pre  = targets["left_pre"],   targets["right_pre"]
    left_gr,   right_gr   = targets["left_grasp"], targets["right_grasp"]
    # Per-arm approach orientation (torso frame) comes from the caller:
    # R_left = Rz(box yaw) keeps the left hand square to the yawed box;
    # R_right = identity keeps the right square to the torso (as validated);
    # its wrist-yaw is locked, so this is a soft bias IK can't fully meet.

    steps = []

    # 0. Optional clearance to a known start pose. CLOSE fires here when present.
    if clearance is not False:
        if clearance is True:
            q_clear = np.concatenate([_REST, _REST_RIGHT])
        else:
            q_clear = np.asarray(clearance, dtype=float)
            if q_clear.shape != (14,):
                raise ValueError(
                    f"clearance must be True/False or a 14-vec; "
                    f"got shape {q_clear.shape}")
        steps.append({
            "label":           "both -> clearance",
            "side":            "both",
            "target_q":        q_clear,
            "obstacles":       obs_full,     # goal checked against box world
            "skip_planning":   True,         # direct joint move; goal-only check
            "gestures_before": _g("close"),
            "settle":          clearance_settle,
        })

    # 1. approach -> pre-grasp, box present. Open once arrived (before close-in).
    #    CLOSE here only if there's no clearance step to carry it.
    approach = {
        "label":            "approach -> pre-grasp",
        "side":             "both",
        "target_xyz_left":  np.asarray(left_pre,  dtype=float),
        "target_xyz_right": np.asarray(right_pre, dtype=float),
        "target_R_left":    R_left,
        "target_R_right":   R_right,
        "obstacles":        obs_full,
        "skip_planning":    False,
        "gestures_after":   _g("open"),
        "settle":           approach_settle,
    }
    if clearance is False:
        approach["gestures_before"] = _g("close")
    steps.append(approach)

    # 2. close in -> grasp, box dropped. Grab once seated.
    steps.append({
        "label":            "close in -> grasp",
        "side":             "both",
        "target_xyz_left":  np.asarray(left_gr,  dtype=float),
        "target_xyz_right": np.asarray(right_gr, dtype=float),
        "target_R_left":    R_left,
        "target_R_right":   R_right,
        "obstacles":        obs_no_box,
        "skip_planning":    True,
        "gestures_after":   _g("grab"),
        "settle":           grasp_settle,
    })

    # 3. lift, box held. Joint-space goal: the grasp pose with both shoulder
    #    pitches rotated by lift_sp_deg — no IK, no Cartesian target. q_start
    #    (the grasp commanded q) is threaded in via chain_start, so this is a
    #    pure relative joint move, continuous with the grasp. SP is index 0 of
    #    each 7-DOF arm -> left SP = q[0], right SP = q[7].

    steps.append({
        "label":         "lift",
        "side":          "both",
        "target_q":       np.concatenate([_LIFT_LEFT, _LIFT_RIGHT]),
        "obstacles":     obs_no_box,
        "skip_planning": True,
        "settle":        lift_settle,
    })

    return steps


# ===========================================================================
# PUT DOWN + CLEAR  (runtime builder — place, release, push the box aside)
# ---------------------------------------------------------------------------
# Sets a held box on the table, releases it, retreats the RIGHT arm, then uses
# the LEFT hand to nudge the box to the robot's right (-y) and back out, re-close
# and plan home. Like build_grab it needs the live obstacle worlds and FKs the
# (joint-space) put-down pose to place the Cartesian moves, so it takes the
# dual-arm model. Assumes the run STARTS with the box held (immediately after a
# build_grab lift) — no initial regrip.
#
# Self-contained: returns all six steps, including the final left-arm return to
# rest. No box re-perception — the left return plans in the box-free world
# (obs_no_box), so the box's post-push location doesn't matter to the planner.
#
#   1 both  lower onto table + OPEN  (open rides step 1's gestures_after)
#   2 right hand out (-x back, -y), hold orientation, re-CLOSE
#   3 right hand plan to rest                                  (RRT, obs_no_box)
#   4 left  hand push box to the right (-y) by push_m   (no orientation target)
#   5 left  hand back out (+y) by back_m                (no orientation target)
#   6 left  hand CLOSE + plan to rest                          (RRT, obs_no_box)
#
# Obstacle worlds:
#   obs_no_box — used by every step here: the box is in hand / just released /
#                being pushed, and the right + left returns plan in the box-free
#                world (we are deliberately maneuvering near the box, and the
#                push moved it anyway).
#   obs_full   — accepted for signature symmetry with build_grab but currently
#                unused (every step uses obs_no_box). Kept so the caller's
#                build_put_down(arm, OBS_FULL, OBS_NO_BOX) line stays stable.
#
# Gesture timing:
#   OPEN  rides step 1's gestures_after — release once the box is down.
#   CLOSE (right) rides step 2's gestures_after — re-fist the right hand once
#         clear, before it plans home. The LEFT hand stays OPEN through the push
#         + back-out (it pushes open-handed) and re-closes on step 6 before it
#         plans home.
#
# Directions (note: +y = robot's LEFT, -y = robot's RIGHT):
#   right hand out  : -x by out_m (back toward the body) and -y by 0.1
#   push            : -y by push_m (left hand on the box's +y face nudges it right)
#   left hand back  : +y by back_m (retreat off the box face)
# Push + back-out are position-only per request (no target_R). If the unlocked
# left wrist drifts during these translations, pin target_R to R_left to hold it.
# ===========================================================================

_PUTDOWN_LEFT = np.radians([
     -26.87,   # shoulder pitch
      +6.05,   # shoulder roll
      +1.72,   # shoulder yaw
     +45.87,   # elbow
      -4.60,   # wrist roll
     -29.32,   # wrist pitch
      -3.37,   # wrist yaw
])
_PUTDOWN_RIGHT = np.radians([
     -19.81,   # shoulder pitch
      -6.79,   # shoulder roll
      -4.79,   # shoulder yaw
     +33.26,   # elbow
      +1.85,   # wrist roll
     -19.15,   # wrist pitch
      -0.67,   # wrist yaw
])


def build_put_down(arm_model, obs_full, obs_no_box, *,
                   push_m=0.15, out_m=0.25, back_m=0.35,
                   lower_settle=1.0,
                   right_out_settle=0.1,
                   right_rest_settle=0.5,
                   push_settle=0.3,
                   back_settle=0.5,
                   rest_settle=0.3,
                   hands=("left", "right")):
    """Return the place-and-clear step list (steps 1-6; see header).

    Parameters
    ----------
    arm_model   : G1DualArm. Used only for FK of the put-down pose, to place
                  the right-hand-out / push / back-out Cartesian targets.
    obs_full    : accepted for signature symmetry with build_grab; currently
                  unused (every step plans in obs_no_box). Kept so the caller's
                  build_put_down(arm, OBS_FULL, OBS_NO_BOX) line stays stable.
    obs_no_box  : (N,9) world WITHOUT the box — used by every step (the moves
                  maneuver next to / against the box, and after the push the box
                  location is stale, so the returns plan in the box-free world).
    push_m      : how far the left hand nudges the box to the right (-y).
    out_m       : right-hand retreat back toward the body (-x) before it plans home.
    back_m      : left-hand retreat (+y) off the box face after the push.
    lower_settle      : dwell after reaching the put-down pose; covers the OPEN sweep.
    right_out_settle  : dwell after the right-hand-out move (covers right CLOSE).
    right_rest_settle : dwell after the right arm reaches rest.
    push_settle       : dwell after the push.
    back_settle       : dwell after the left-hand back-out.
    rest_settle       : dwell after the left arm reaches rest (step 6).
    hands       : which hands the OPEN at step 1 drives (right CLOSE is explicit).

    The put-down pose is the tuned joint-space target (_PUTDOWN_LEFT/RIGHT).
    Assumes the box is already held (run right after build_grab's lift); the
    first step is a direct joint move continuous with the lift, so the grip
    carries over with no regrip. Self-contained: the final step is the left-arm
    return to rest (no separate return builder, no box re-perception).
    """
    def _g(name):
        return {s: name for s in hands}

    q_pd_left  = _PUTDOWN_LEFT
    q_pd_right = _PUTDOWN_RIGHT

    # FK the put-down pose: EE position + orientation per hand. The right-out
    # step holds R_right; push / back-out are position-only (no target_R).
    T_left  = arm_model.left.forward_kinematics(q_pd_left)[-1]
    T_right = arm_model.right.forward_kinematics(q_pd_right)[-1]
    ee_left            = T_left[:3, 3]
    ee_right, R_right  = T_right[:3, 3], T_right[:3, :3]

    right_out = ee_right + np.array([-out_m, -0.1,  0.0])  # right hand out (-x back, -y)
    push_to   = ee_left  + np.array([0.0, -push_m, 0.0])   # nudge box right (-y)
    back_to   = push_to  + np.array([-0.4, +back_m, 0.0])   # left hand back out (+y)

    steps = []

    # 1. Lower the held box onto the table (joint-space, continuous from lift).
    #    Release once down — OPEN (both hands) rides gestures_after. Box in hand
    #    / just released -> obs_no_box.
    steps.append({
        "label":          "lower box -> put down",
        "side":           "both",
        "target_q":       np.concatenate([q_pd_left, q_pd_right]),
        "obstacles":      obs_no_box,
        "skip_planning":  True,
        "gestures_after": _g("open"),
        "settle":         lower_settle,
    })

    # 2. Right hand out (-x back, -y), clear of the box, holding orientation.
    #    Re-fist the RIGHT hand once clear (CLOSE rides gestures_after).
    steps.append({
        "label":          "right hand out (clear box)",
        "side":           "right",
        "target_xyz":     right_out,
        "target_R":       R_right,
        "obstacles":      obs_no_box,
        "skip_planning":  True,
        "gestures_after": {"right": "close"},
        "settle":         right_out_settle,
    })

    # 3. Right arm plans back to rest (box-free world).
    steps.append({
        "label":         "right hand plan to rest",
        "side":          "right",
        "target_q":      _REST_RIGHT,
        "obstacles":     obs_no_box,
        "skip_planning": False,
        "settle":        right_rest_settle,
    })

    # 4. Left hand nudges the box to the robot's right (-y). Position-only (no
    #    orientation target). obs_no_box — we are deliberately in contact.
    steps.append({
        "label":         "left hand push box right (-y)",
        "side":          "left",
        "target_xyz":    push_to,
        "obstacles":     obs_no_box,
        "skip_planning": True,
        "settle":        push_settle,
    })

    # 5. Left hand backs out (+y) off the box face. Position-only, obs_no_box.
    steps.append({
        "label":         "left hand back out (+y)",
        "side":          "left",
        "target_xyz":    back_to,
        "obstacles":     obs_no_box,
        "skip_planning": True,
        "settle":        back_settle,
    })

    # 6. Left hand re-closes and plans back to rest, in the box-free world
    #    (obs_no_box) — no box re-perception needed. CLOSE rides gestures_before
    #    so the fist forms before the retract.
    steps.append({
        "label":           "left close + plan to rest",
        "side":            "left",
        "target_q":        _REST,
        "obstacles":       obs_no_box,
        "skip_planning":   False,
        "gestures_before": {"left": "close"},
        "settle":          rest_settle,
    })

    return steps


# ===========================================================================
# PULL LEFT  (runtime builder — drag a box jammed on the right toward +y)
# ---------------------------------------------------------------------------
# For a box pushed up against the wall on the robot's right, too far to grasp:
# the RIGHT hand closes, plans to a staging pose behind/below/left of the box
# tag, then drags the box toward +y (the robot's left) by drag_m. Stops after
# the drag — no regrip and no hand extraction yet (that's a follow-up).
#
#   1     right hand CLOSE + plan to staging pose          (RRT, obs_full)
#   2..N  drag +y in equal position-only sub-steps         (direct, obs_no_box)
#
# Staging pose = box-tag origin + a base-frame offset (NOT tag-frame): behind
# the tag (-x), to its left (+y), and below it (-z), with orientation identity.
# The drag is split into drag_substeps equal +y moves so the end-effector tracks
# a near-straight Cartesian line; a single long joint-space interp between two IK
# solutions would bow the path and risk dragging the box into the table or
# lifting it. The sub-steps carry no settle between them, so run_sequence batches
# them into one smooth streamed trajectory.
#
# Obstacle worlds:
#   obs_full   — box present; the approach routes around it to get behind/below.
#   obs_no_box — in contact, dragging; the box moves with the hand so it is not
#                an obstacle. (Walls are NOT modeled — the whole reason to drag.)
#
# Caveats (flagged, not handled here):
#   * The approach targets orientation identity while the right wrist-yaw is
#     LOCKED (broken joint). If identity isn't achievable with the wrist at its
#     pre-positioned angle, IK settles with orientation error that preflight
#     (limits + collision only) won't flag — watch the approach on hardware.
#   * The drag is position-only (no target_R per request), so wrist orientation
#     may drift across the 30 cm. Pin target_R on the drag steps if the hook
#     disengages.
# ===========================================================================

# Base-frame offset from the box tag to the staging pose: -x back, +y left,
# -z below (5 cm left, 6 cm below, 15 cm back).
_PULL_OFFSET = np.array([-0.20, +0.05, +0.005])


def build_pull_left(arm_model, T_base_box, obs_full, obs_no_box, *,
                    drag_m=0.30, drag_substeps=6,
                    approach_settle=0.5, drag_settle=0.5,
                    R_approach=None):
    """Return the pull-left step list (see header). Stops after the drag.

    Parameters
    ----------
    arm_model     : G1DualArm (only the right sub-arm is driven).
    T_base_box    : 4x4 box-tag pose in the base frame; the tag origin is
                    T_base_box[:3, 3]. The staging pose is tag + _PULL_OFFSET.
    obs_full      : (N,9) world WITH the box — for the approach RRT.
    obs_no_box    : (N,9) world WITHOUT the box — for the drag (in contact).
    drag_m        : total +y drag distance (default 0.30 m).
    drag_substeps : number of equal +y sub-moves the drag is split into, so the
                    EE follows a near-straight line (default 6 -> 5 cm each).
    approach_settle : dwell after reaching the staging pose.
    drag_settle     : dwell after the final drag sub-step.
    R_approach    : orientation target for the approach (default identity).
    """
    if R_approach is None:
        R_approach = np.eye(3)
    tag = np.asarray(T_base_box, dtype=float)[:3, 3]
    approach = tag + _PULL_OFFSET
    reach_in   = approach  + np.array([0.04, 0.0, 0.0])

    steps = []

    # 1. Right hand close + plan to the staging pose (orientation identity).
    #    Box present -> obs_full so the approach routes around it.
    steps.append({
        "label":           "pull: right -> staging (behind/below box)",
        "side":            "right",
        "target_xyz":      approach,
        "target_R":        R_approach,
        "obstacles":       obs_full,
        "skip_planning":   False,
        "gestures_before": {"right": "close"},
        "gestures_after": {"right": "point"},
        "settle":          approach_settle,
    })

    steps.append({
        "label":           "pull: right -> reach in box",
        "side":            "right",
        "target_xyz":      reach_in,
        "obstacles":       obs_no_box,
        "skip_planning":   True,
        "settle":          0.3,
    })
    # 2..N. Drag +y in equal position-only sub-steps (no orientation target),
    #    in contact with the box -> obs_no_box. No settle between sub-steps, so
    #    run_sequence batches them into one smooth straight-line drag.
    '''
    for k in range(1, drag_substeps + 1):
        d = drag_m * k / drag_substeps
        steps.append({
            "label":         f"pull: drag +y {d*100:.0f}cm",
            "side":          "right",
            "target_xyz":    reach_in + np.array([0.0, d, 0.0]),
            "obstacles":     obs_no_box,
            "skip_planning": True,
            "settle":        drag_settle if k == drag_substeps else 0.0,
        })
    
    steps.append({
        "label":         f"pull: drag +y 30cm",
        "side":          "right",
        "target_xyz":    reach_in + np.array([0.0, drag_m, 0.0]),
        "obstacles":     obs_no_box,
        "skip_planning": False,
        "settle":        drag_settle,
    })
    '''

    return steps


# ---------------------------------------------------------------------------
# Static choreography registry — names runnable via g1_plan_run --choreography.
# Empty now: the demo choreographies (waves) were removed as off-task. The grasp
# behaviors are runtime-built (build_grab / build_put_down / build_pull_left) and
# are NOT static entries. Kept as an (empty) dict so g1_plan_run's import of
# CHOREOGRAPHIES stays valid; add canned, perception-free sequences here if you
# ever want them on the --choreography CLI again.
# ---------------------------------------------------------------------------

CHOREOGRAPHIES = {}