# G1 Vision-Guided Grasping — Development Doc

End-to-end pipeline that lets the G1 humanoid use its RealSense D435i + two
AprilTags (one on the box, one on the table) to locate a box and perform a
dual-arm side-pinch manipulation:

- **grab + lift** — approach → close in → grip → lift
- **put down + push** — place the box, release, and nudge it to the robot's right
- **pull left** — for a box jammed on the robot's right, drag it back toward the
  left (drag step currently staged but disabled — see notes)

There are exactly **two entry points**:

| Run this | To do this |
|---|---|
| `python grab_run.py` | grab → lift → (optional) put-down + push right |
| `python pull_left_run.py` | pull a far-right box back toward the left |

If you just want to see the robot grab the box, jump to
[Running the system](#running). The rest is reference for when something needs
changing — especially [Tuning](#tuning), which lists every parameter, where it
lives, and how to dial it.

---

## Table of contents

1. [Architecture](#architecture)
2. [Hardware assumptions](#hardware-assumptions)
3. [File reference — what each file does](#files)
4. [Running the system](#running)
5. [The locked wrist joint (RWY)](#locked-joint)
6. [The choreography layer](#choreography-layer)
7. [Tuning — every knob, where it lives, how to dial it](#tuning)
8. [One-time setup (Docker image for perception)](#one-time-setup)
9. [Camera-height calibration procedure](#calibration)
10. [Known issues / future work](#known-issues)
11. [Troubleshooting](#troubleshooting)

---

## Architecture <a name="architecture"></a>

Two Python processes on the same Jetson, talking over TCP on `localhost:5000`.
**Why two processes?** `pyrealsense2` is compiled against Python 3.8 and won't
import into the conda 3.12 env where the rest of the stack lives. A socket
sidesteps the Python-version wall.

```
┌─────────────────────────────────────────────────────────────────┐
│ HOST: humanoid (Jetson Orin NX, Ubuntu 20.04, aarch64)          │
│                                                                 │
│  ┌──────────────────────────────────────────┐                   │
│  │ Docker container: realsense:pyrs2 (3.8)  │                   │
│  │   apriltag_pose_server.py                │                   │
│  │     pyrealsense2 -> BGR frame + K, D     │                   │
│  │     pupil_apriltags -> tag corners       │                   │
│  │     cv2.solvePnP -> T_optical_tag (4x4)  │                   │
│  │     TCP server on 0.0.0.0:5000           │  one frame =       │
│  └──────────────────┬───────────────────────┘  per-tag 4x4      │
│                     │ socket                                     │
│  ┌──────────────────▼───────────────────────┐                   │
│  │ Conda base (Python 3.12)                 │                   │
│  │   perception_client.py  (bg thread)      │                   │
│  │   camera_calib.optical_to_base           │                   │
│  │   grasp_target / grasp_scene             │  tag -> box/table  │
│  │       -> box pose, grasp targets,        │  -> targets +      │
│  │          2 obstacle worlds (Scene)       │     obstacle worlds│
│  │   G1DualArm.inverse_kinematics  (14-DOF) │                   │
│  │   g1_plan_run.plan (RRT-Connect+shortcut)│                   │
│  │   g1_choreograph.run_sequence            │  plan + stream     │
│  │   G1ArmController / G1HandController      │                   │
│  └────────────┬─────────────────────────────┘                   │
│               │ unitree_sdk2py / CycloneDDS                      │
│               ▼ /rt/arm_sdk                                      │
└───────────────────────────────────────────────────────────────────┘
```

**Data flow (one grab):**

```
camera -> AprilTag detect -> solvePnP -> T_optical_tag
       -> optical_to_base  -> T_base_tag
       -> box_in_base / table_in_base -> box & table poses (yaw-only)
       -> grasp_targets() -> left/right pre + grasp xyz
       -> make_box_obb / make_table_slab -> obs_full (box present)
                                            obs_no_box (box dropped)
       -> build_grab() -> [clearance, approach, close-in, lift] steps
       -> run_sequence() -> per step: snap locked joint -> IK -> RRT -> stream,
                            interleaved with hand gestures (close/open/grab)
       -> box gripped & lifted -> (optional) build_put_down() -> place + push
```

All of `snapshot_scene`'s outputs are produced **once** at the top of a run and
frozen in a `Scene` object (see `grasp_scene.py`).

---

## Hardware assumptions <a name="hardware-assumptions"></a>

- **G1 humanoid** with dex3 hands, accessed via `unitree_sdk2py`.
- **RealSense D435i** USB-mounted to the G1 neck. All neck USB ports are
  **USB 2.0** — the stream config must respect that bandwidth (640×480 @ 6 FPS).
  Verify with `lsusb | grep Intel` on the host.
- **Right wrist yaw (RWY) is mechanically locked** — see
  [the locked joint](#locked-joint). Pre-position it by hand before every run.
- **Two AprilTags, family `tag36h11`:**
  - **id 0** on the box top face, **3.4 cm**. The tag's bottom-left corner sits
    at the geometric center of the top face; tag `+y` points away from the robot.
  - **id 1** on the table (far-side middle), **8 cm**, lying flat.
- **Box dimensions:** 28 × 30 × 12 cm (x = depth, y = width, z = height in box
  frame). The 30 cm dimension is parallel to G1's lateral axis (the pinch axis).

---

## File reference — what each file does <a name="files"></a>

### Entry points (run these)

- **`grab_run.py`** — THE main grasp deliverable. Connects the arm + both hands,
  snapshots the scene, validates IK, builds the grab choreography, runs it behind
  a single confirm, then offers a put-down (place + push aside). Reads
  top-to-bottom: `snapshot_scene → solve_and_validate → build_grab → run_sequence
  → build_put_down`.
- **`pull_left_run.py`** — the pull-left behavior, separated out. Same front-end,
  but runs `build_pull_left` (right hand to a box jammed on the right) and stops.

### Perception

- **`apriltag_pose_server.py`** — runs **inside the Docker container**. Reads
  RealSense frames, detects all `tag36h11` tags in one pass, solves PnP per tag
  (sizes from the per-ID `TAG_SIZES` map), and serves each `T_optical_tag` over
  TCP. Resilient to USB hiccups (catches frame timeouts, restarts the pipeline
  after 3 misses).
- **`perception_client.py`** — `PerceptionClient`: a background-threaded TCP
  client (daemon thread). `.start()` then `.latest_pose(tag_id)` returns the most
  recent 4×4 for that tag, or `(None, None)` if stale (>1 s old). Never blocks the
  caller on the network.
- **`camera_calib.py`** — `optical_to_base(T_optical_tag)` applies the fixed
  camera extrinsic `T_BASE_OPTICAL` (mount pitch + translation, then the optical→
  body axis swap). This is where the camera mount numbers live.

### Scene + grasp geometry

- **`grasp_target.py`** — box/table geometry. `box_in_base` / `table_in_base`
  chain the tag pose into base frame and **zero pitch/roll** (flat surfaces, keep
  only yaw). `grasp_targets()` returns the five base-frame xyz points
  (box_center, left/right pre, left/right grasp). Holds the box/table dimensions
  and the tag-on-object orientation conventions (`R_box_tag`, `R_table_tag`).
- **`grasp_scene.py`** — the shared front-end for both runners. `snapshot_scene
  (client)` waits for both tags and returns a frozen `Scene` (poses, grasp
  targets, `obs_full`/`obs_no_box`, `names_for`). Also holds `make_box_obb`,
  `make_table_slab`, `solve_and_validate` (IK + validate + named collision
  report), `report_ee`, and `render_planned` (post-run plotly HTML). Everything
  the runners share lives here — they import from here, not from each other.

### Kinematics + planning

- **`myArm_v2.py`** — `G1Arm`: single-arm kinematics over the 7-DOF arm chain
  (ikpy under the hood), FK, IK, capsule-vs-OBB collision checking, joint limits.
- **`myArm_dual.py`** — `G1DualArm`: the 14-DOF dual-arm model. Owns `.left` and
  `.right` `G1Arm`s, the combined IK (`inverse_kinematics`), collision
  (`check_collision`, `collision_report`), and the locked-joint machinery.
- **`rrt_planner.py`** / **`kdtree.py`** — RRT-Connect and its nearest-neighbor
  KD-tree. Called by `plan()`.
- **`g1_plan_run.py`** — home of the planning helpers the whole stack imports:
  `plan()` (RRT-Connect + shortcut), `_validate()` (goal reachable/in-limits/
  collision-free), `_default_obstacles()` (the static torso box), and
  `save_path_html()` (plotly viz). It *also* has a general-purpose CLI `main()`
  (`--simonly`, `--target`, etc.) that is **not used by the grasp task** — it's a
  separate hardware-free mover, kept for ad-hoc moves and sim planning.
- **`gravity_comp.py`** — `GravityComp`: Pinocchio-based gravity feed-forward
  torque, scaled 0–1.

### Choreography

- **`g1_choreograph.py`** — the engine. `run_sequence(robot, arm, steps,
  confirm=, record=)` plans + executes a list of step dicts (batching compatible
  consecutive steps into one smooth spline), threading each group's commanded
  endpoint into the next as the IK seed. `preflight_validate(arm, steps)` dry-
  checks every goal without moving. Owns the locked-joint snap (`_snap_locked`)
  and the per-step obstacle-world handling. Full step schema is in its docstring.
- **`choreographies.py`** — the three runtime builders: `build_grab`,
  `build_put_down`, `build_pull_left`, plus the shared poses they use (`_REST`,
  `_REST_RIGHT`, `_LIFT_LEFT`, `_LIFT_RIGHT`). `CHOREOGRAPHIES` is an empty
  registry (the old demo waves were removed).

### Hardware layer

- **`g1_robot.py`** — `G1Robot`: umbrella context manager owning the arm + both
  hand controllers, with a deterministic safe-disconnect on exit (`with G1Robot(
  ...) as robot:`).
- **`g1_arm_controller.py`** — `G1ArmController`: streams trajectories over
  CycloneDDS, runs the 100 Hz PD loop, masks locked joints from the tracking-
  error abort.
- **`g1_hand_controller.py`** — `G1HandController`: the dex3 hands, 100 Hz loop
  holding the last commanded gesture. `GESTURES_RAD` defines named poses
  (`open`, `close`, `grab`, `point`, `peace`).
- **`g1_arm_live_tuner.py`** — provides joint-index constants (`G1JointIndex`,
  `JOINT_NAME`, `ARM_JOINTS`, …) and a `LiveTuner`. **Required by
  `g1_arm_controller`** (despite the "tuner" name), so don't delete it; also
  runnable standalone for live PD/angle tuning.

### Other

- **`g1_29dof_with_hand.urdf`** — the robot model loaded by `myArm_v2` and
  `gravity_comp`. The only URDF the stack uses.

---

## Running the system <a name="running"></a>

Two terminals on the host (humanoid), both via VS Code Remote-SSH.

### 0. Pre-position the broken wrist (every run)

The right wrist yaw motor can't move (see [locked joint](#locked-joint)). Before
starting, **physically rotate the right wrist** to a natural angle for the grasp.
The script reads that angle at startup and plans around it.

### 1. Terminal 1 — perception server (Docker)

```bash
docker run -it --rm --network host --privileged -v /dev:/dev \
  -v $(pwd):/work realsense:pyrs2 bash
python3 /work/apriltag_pose_server.py     # leave running
```

Expect `[camera] 640x480@6 …`, `[socket] listening on 0.0.0.0:5000`, and a
`T_optical_tag` print whenever a tag is in view. It prints `.` when no tag is
seen — that's normal, not a stall.

### 2. Terminal 2 — the grab (host conda)

```bash
conda activate base
cd ~/Installed_Libraries/unitree_sdk2_python/example/g1/high_level/raymond_ws
python grab_run.py
```

What `grab_run.py` does, start to finish:

1. Connects arm + both dex3 hands with gravity comp.
2. Reads the measured start pose and the live RWY lock angle; builds
   `G1DualArm(locked_joints_right={6: rwy_lock})`.
3. `snapshot_scene()` waits up to 10 s for **both** tags (box id 0 + table id 1)
   and builds the box/table poses, the four grasp targets, and the two obstacle
   worlds. Aborts if either tag is missing.
4. Validates IK for pre-grasp and grasp (reachable, in-limits, collision-free in
   their worlds) — aborts before any motion if a target is unreachable.
5. Builds the grab choreography and preflights every step (still no motion).
6. Asks **once** `Start grab? …`. On `y`, runs all phases automatically:
   clearance → approach → close-in (grip) → lift.
7. Renders the planned (RRT) legs to plotly HTML *after* the run (no extra
   planning cost), then reports end-effector error.
8. If the lift succeeded, asks **once** `Start PUT-DOWN …`. On `y`: place,
   release, retreat the right arm, push the box right with the left hand, back
   out, and plan both arms home.
9. If you decline put-down, it holds the grip and waits for Enter before
   releasing (so the lifted box doesn't drop the instant the script exits).

### 2b. Terminal 2 — the pull-left

```bash
python pull_left_run.py
```

Same front-end, then one `Run PULL-LEFT …?` gate. On `y`, the right hand closes,
plans to a staging pose behind/below/left of the box tag, and reaches in.

> **Note:** the actual +y drag is **intentionally commented out** in
> `build_pull_left` (`choreographies.py`) — the hand-extraction strategy after
> the drag isn't settled yet. As shipped, pull-left approaches and reaches in,
> then stops. Re-enable the drag block in `build_pull_left` when ready.

---

## The locked wrist joint (RWY) <a name="locked-joint"></a>

The **right wrist yaw** motor (RWY) is physically broken: the encoder reads, but
the motor cannot actuate. The stack handles this in three places — understand all
three before changing the planning path.

1. **Read the live angle at startup.** RWY is joint **index 6** in the right
   7-DOF arm, which is index **13** in the 14-DOF dual vector:
   ```python
   q_start  = ctrl.get_current_q(side="both")
   rwy_lock = float(q_start[13])
   arm = G1DualArm(locked_joints_right={6: rwy_lock})
   ```
   This clamps RWY's joint limits to a **zero-width interval** `[lock, lock]`, so
   IK and collision treat it as fixed at whatever you pre-positioned it to.

2. **Snap before every plan (`_snap_locked`).** Any `q` whose RWY value carries a
   slightly different floating-point tail (from a measured readback, a chained
   endpoint, or IK) reads as *out of limits* against that zero-width interval and
   stalls RRT at step 0 — the classic **"5000 iterations, no path"** that looks
   like a planner bug but isn't. `run_sequence` defends against this by snapping
   locked joints onto their exact bit-identical lock value before every plan and
   limits check. Add more locked joints later and this handles them automatically.

3. **Mask it from the tracking-abort.** A locked motor can't track a commanded
   trajectory, so the executor would otherwise abort on its tracking error. The
   runner tells the controller to ignore it:
   ```python
   robot.arm.locked_right = list(arm.right.locked_joints.keys())
   ```

**Practical consequence for tuning:** the right hand's orientation is bounded by
wherever you pre-positioned the wrist. The grab targets the right hand with
**orientation identity** (`R_right = np.eye(3)`) as a soft bias the IK can't fully
meet; the left hand tracks the box yaw (`R_left = Rz(box yaw)`). If the right
hand sits at a bad angle, re-position the wrist physically — there is no software
fix.

---

## The choreography layer <a name="choreography-layer"></a>

A **step** is a dict describing one move; a **sequence** is a list of steps.
`run_sequence` plans and executes them in order. The grasp behaviors are built at
runtime (the targets come from the live tag) by `build_grab` / `build_put_down` /
`build_pull_left` in `choreographies.py`.

**A step's fields** (full spec in `g1_choreograph.py`):

- `side` — `"left"`, `"right"`, or `"both"` (required).
- One target form: `target_q` (direct joint angles), `target_xyz` (+ optional
  `target_R`) for one arm, `target_xyz_left`/`_right` (+ `target_R_left`/`_right`)
  for both, or `q_delta` (relative joint goal `q_goal = q_start + q_delta`, no IK).
- `gestures_before` / `gestures_after` — `{"left": name, "right": name}`, fire a
  hand gesture before/after the move.
- `settle` — seconds to dwell after the move.
- `skip_planning` — straight joint move (goal still collision-checked; the swept
  path is not RRT'd).
- `obstacles` — a per-step obstacle world. The grab uses **two**: `obs_full`
  (box present) for the approach, `obs_no_box` for grasp/lift (the hands are
  meant to touch the box). Steps with different worlds are never batched.

`preflight_validate` dry-checks every goal without moving. `run_sequence(...,
confirm=cb)` calls `cb(group_index, num_groups, group)` before each group; the
runners use it to prompt once and then auto-run.

---

## Tuning — every knob, where it lives, how to dial it <a name="tuning"></a>

This is the single source for what to set and how. Grouped by what they affect.

### Perception / tags — `apriltag_pose_server.py`

| Param | Default | How to tune |
|---|---|---|
| `TAG_SIZES` | `{0: 0.034, 1: 0.08}` | Physical **inner black-square** edge per tag id, in meters. **Most error-prone value** — a wrong size scales that tag's reported distance proportionally. Measure with a ruler. |
| `RES_W, RES_H, FPS` | 640×480 @ 6 | USB-2.0-safe. Bump FPS toward 15 if motion blur hurts detection; drop back if you see `Frame didn't arrive within …`. |
| `TAG_FAMILY` | `tag36h11` | The family both tags use. |
| `SOCKET_PORT` | 5000 | Must match the client. |

### Camera extrinsic — `camera_calib.py`

| Param | Default | How to tune |
|---|---|---|
| `CAMERA_MOUNT_XYZ` | `(0.0476, 0.0325, 0.4027)` m | Camera position in base. **The z component is the usual culprit** when grasp positions look off vertically — see [calibration](#calibration). |
| `CAMERA_PITCH_DEG` | 42.4 | Downward camera tilt (positive = nose down). If a tag below the mount shows up *above* it, flip the sign. |

### Box + table geometry — `grasp_target.py`

| Param | Default | How to tune |
|---|---|---|
| `BOX_X_LEN, BOX_Y_LEN, BOX_Z_LEN` | 0.28 / 0.30 / 0.12 m | Physical box dims (x depth, y width, z height). y is the pinch axis. |
| `R_box_tag` | (current) | Tag-on-box orientation. Update if the box tag is mounted differently (tag +y → box +x, tag +x → box −y). |
| `TABLE_THICK` | 0.04 m | Slab thickness of the table obstacle. |
| `TABLE_WIDTH` | 2.0 m | y-extent (treated as "very wide"; not located from the tag). |
| `TAG_TO_FAR_EDGE` / `TAG_TO_NEAR_EDGE` | 0.09 / 0.517 m | Table-tag corner → far / near edge along table +x. Sets `TABLE_DEPTH`. |
| `R_table_tag` | (current) | Tag-on-table orientation. **Verify physically** — one sign flip corrects a mirrored mount. |

### Grasp targets — `grasp_target.grasp_targets()`

These live as locals inside `grasp_targets()` (all box-frame offsets, meters):

| Param | Default | How to tune |
|---|---|---|
| `clearance` (arg) | 0.15 | How far pre-grasp sits outside the box faces along y. Increase if leg-1 RRT struggles; decrease for snappier motion. |
| `GRASP_X_OFFSET` | −0.09 | Grasp position along box depth (0 = center). Negative = toward the robot. |
| `GRASP_Z_OFFSET` | 0.00 | Grasp height within the box (+ = toward the top face). Raise to lift the whole approach corridor off the table slab if RRT fights the table. **Don't** use it to compensate for camera-z errors. |
| `HAND_PENETRATION` | −0.01 | Fingertip depth past the box face (+ = into the box). Compensates finger compliance / wrap-around. |
| `pregrasp_z_lift` (arg) | 0.0 | Pre-grasp height above the grasp, so close-in descends as it closes. |

### Grab + lift — `grab_run.py` + `build_grab` (`choreographies.py`)

| Param | Where | Default | How to tune |
|---|---|---|---|
| `MAX_ARM_SPEED_DPS` | `grab_run.py` | 80 | Slew cap. Start low for first bring-up; ramp after validation and re-check the tracking-abort threshold. |
| `LIFT_SP_DEG` | `grab_run.py` | −30 | Shoulder-pitch delta at the lift (a **commanded joint** delta, not a Cartesian height; negative raises). Start small, increase. |
| `GRAVITY_COMP_SCALE` | `grab_run.py` | 1.0 | Gravity feed-forward scale (0–1). |
| `_REST`, `_REST_RIGHT` | `choreographies.py` | (tuned) | Clearance/idle poses, degrees. The approach plans from `_REST`/`_REST_RIGHT`. |
| `_LIFT_LEFT`, `_LIFT_RIGHT` | `choreographies.py` | (tuned) | The lift target pose, degrees. |
| `approach_settle` | `build_grab` | 0.3 | Dwell at pre-grasp; must cover the open-fist sweep. Raise (≥ ~2 s) if hands aren't fully open before close-in. |
| `clearance_settle` / `grasp_settle` / `lift_settle` | `build_grab` | 0.1 / 0.2 / 0.1 | Per-phase dwell. |

### Put-down + push right — `build_put_down` (`choreographies.py`)

Directions: **+y = robot's LEFT, −y = robot's RIGHT**. Push moves the box −y.

| Param | Default | How to tune |
|---|---|---|
| `_PUTDOWN_LEFT`, `_PUTDOWN_RIGHT` | (tuned) | The place pose (joint-space, degrees); FK'd to seed the push/retreat Cartesian moves. |
| `push_m` | 0.15 m | How far the left hand nudges the box right (−y). Open-loop contact — confirm the box actually shifts on hardware. |
| `out_m` | 0.25 m | Right-hand retreat back toward the body (−x) before it plans home. |
| `back_m` | 0.35 m | Left-hand retreat (+y) off the box face after the push. |
| `lower_settle` | 1.0 | Dwell after placing; covers the OPEN release sweep. |
| `right_out_settle` / `right_rest_settle` / `push_settle` / `back_settle` / `rest_settle` | 0.1 / 0.5 / 0.3 / 0.5 / 0.3 | Per-step dwell. |

### Pull-left — `pull_left_run.py` + `build_pull_left` (`choreographies.py`)

| Param | Where | Default | How to tune |
|---|---|---|---|
| `MAX_ARM_SPEED_DPS` / `GRAVITY_COMP_SCALE` | `pull_left_run.py` | 80 / 1.0 | As above. |
| `DRAG_M` | `pull_left_run.py` | 0.30 | Total +y drag distance — **only takes effect once the drag block in `build_pull_left` is re-enabled.** |
| `_PULL_OFFSET` | `build_pull_left` | `[−0.20, +0.05, +0.005]` m | Base-frame offset from the box tag to the staging pose (−x back, +y left, +z slight). |
| `approach_settle` / `drag_settle` | `build_pull_left` | 0.5 / 0.5 | Dwell after staging / after the drag. |

---

## One-time setup (Docker image for perception) <a name="one-time-setup"></a>

The lab's `realsense:r35.3.1-python` image does **not** ship `pyrealsense2`
(despite the name — it's a build environment). Build it once and commit a new
image `realsense:pyrs2`. Skip this if `docker images | grep pyrs2` already shows it.

```bash
# Start a build container (no --rm so we can commit later)
docker run -it --name rs_build --network host realsense:r35.3.1-python bash

# Inside the container:
apt-get update && apt-get install -y git cmake build-essential pkg-config \
  libusb-1.0-0-dev libssl-dev
cd /tmp
git clone --depth 1 -b v2.54.2 https://github.com/IntelRealSense/librealsense.git
cd librealsense && mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=true \
  -DPYTHON_EXECUTABLE=/usr/local/bin/python3 \
  -DFORCE_RSUSB_BACKEND=true -DBUILD_EXAMPLES=false \
  -DBUILD_GRAPHICAL_EXAMPLES=false -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) && make install && ldconfig
# librealsense quirk: .so lands in /usr/local/OFF/ — copy it where Python looks:
cp /usr/local/OFF/pyrealsense2.cpython-38-aarch64-linux-gnu.so* \
   /usr/local/lib/python3.8/dist-packages/
ldconfig
python3 -c "import pyrealsense2 as rs; print(rs.__version__)"   # expect 2.54.2

# pip deps (image's default index is unreachable — use public pypi):
pip install --index-url https://pypi.org/simple/ --upgrade "numpy>=1.20"
pip install --index-url https://pypi.org/simple/ pupil-apriltags
```

Then, in a **separate host terminal** (don't exit the build container):

```bash
docker commit rs_build realsense:pyrs2
```

Back in the container: `exit`, then on the host `docker rm rs_build`.

---

## Camera-height calibration procedure <a name="calibration"></a>

`CAMERA_MOUNT_XYZ` / `CAMERA_PITCH_DEG` are Unitree's nominal numbers and may not
match this specific G1. The **z** component matters most: wrong z makes the hands
target the wrong vertical position.

**Fastest (visual):** put the box at its expected grasp spot, run `grab_run.py`,
confirm the first stage, and visually estimate how far above/below the actual box
the hands ended up. Subtract that delta from `CAMERA_MOUNT_XYZ[2]`, rerun, iterate
2–3 times until under ~2 cm. (`python camera_calib.py` prints the base-frame tag
position if you'd rather compare against a tape-measured truth.)

This is a single-distance linear correction. Multi-distance work (e.g. walking
then grasping) would need proper hand-eye calibration at several poses.

---

## Known issues / future work <a name="known-issues"></a>

- **Pull-left drag is disabled.** `build_pull_left` approaches + reaches in, then
  stops; the +y drag block is commented out until the post-drag hand-extraction
  is designed. Re-enable when ready.
- **Right-hand orientation is bounded by the locked wrist.** See
  [locked joint](#locked-joint) — software can't fix a bad pre-position.
- **Push is open-loop.** The left hand nudges the box with no contact feedback;
  whether it shifts a clean `push_m` is contact-dependent. Nothing downstream
  depends on the box's new position (all returns plan in `obs_no_box`).
- **Mount-z vs `GRASP_Z_OFFSET` are not interchangeable.** `CAMERA_MOUNT_XYZ[2]`
  moves both grasp targets *and* the box obstacle; `GRASP_Z_OFFSET` moves only the
  grasp. Tune mount-z to physical truth, `GRASP_Z_OFFSET` for ergonomics only.
- **Box on a flat table only.** `box_in_base` / `table_in_base` zero pitch/roll.
- **Walking is out of scope** — that's the locomotion (Foxy) box's job, and a
  pick-walk-place needs a grip handoff that doesn't exist yet (the hand loop
  releases on disconnect).
- **Multi-distance hand-eye calibration** and an **inter-arm self-collision
  matrix** are the next obvious robustness items. Lidar/Octomap integration is on
  the horizon and may eventually motivate revisiting a MoveIt migration — discuss
  with the mentor before any architectural pivot.

---

## Troubleshooting <a name="troubleshooting"></a>

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: pyrealsense2` | Wrong image, or the `cp` step skipped | Verify `realsense:pyrs2` exists; redo the build section. |
| `Frame didn't arrive within …` | USB-2.0 bandwidth / hiccup | Lower `FPS` (6→4); check `rs-enumerate-devices` in the container. The server auto-restarts after 3 misses; persistent = physical (USB power/autosuspend). |
| `[scene] no box tag (id 0)` / `no table tag (id 1)` | Tag out of view / too far / wrong id | Both tags must be visible. At ~55 cm a 3.4 cm tag is marginal — closer / steadier / bigger tag. |
| Reported distance off by 30–50% | `TAG_SIZES[id]` doesn't match the printed tag | Measure the black-square edge; update the per-id size. |
| Arm clearly above/below the box | `CAMERA_MOUNT_XYZ[2]` off | See [calibration](#calibration). |
| RRT runs forever / "5000 iterations, no path" on execute but the dry plan worked | Locked RWY value carries FP drift → start reads out-of-limits | Handled by `_snap_locked`; confirm the lock angle was passed into `G1DualArm(locked_joints_right=…)` and you're on the current `g1_choreograph.py`. |
| `[ABORT] no path to pre-grasp` | Box too close, or clearance too small | Increase `clearance` (pre-grasp), or move robot/box; check `q_start` isn't already in collision. |
| `[ABORT] … preflight failed` | A goal is unreachable / in collision in its world | Read the printed pos error + named collision pair; re-check reach / `clearance` / box distance. |
| Hands not fully open at the box | `approach_settle` too short for the open-fist sweep | Raise `approach_settle` in `build_grab` (≥ ~2 s). |
| Box drops the instant the script ends | Expected — exiting `G1Robot` releases the hands | `grab_run` waits for Enter before releasing; keep the process alive while gripping. |
| `'float' object has no attribute 'compute'` | `gravity_comp` passed a float instead of a `GravityComp` | Pass `gravity_comp=GravityComp(...)`; the `scale` (0–1) lives inside it. |
| Container can't reach host over the socket | Started without `--network host` | Always use `--network host`. |

---

## Quick start (TL;DR)

```bash
# 0. Pre-position the right wrist (RWY) by hand.

# Terminal 1 (Docker): perception
docker run -it --rm --network host --privileged -v /dev:/dev \
  -v $(pwd):/work realsense:pyrs2 bash
python3 /work/apriltag_pose_server.py        # leave running

# Terminal 2 (host conda): the grab (then optional put-down + push)
conda activate base
cd ~/Installed_Libraries/unitree_sdk2_python/example/g1/high_level/raymond_ws
python grab_run.py

# or, for a box jammed on the right:
python pull_left_run.py
```

---

*Layout: two entry points (`grab_run.py`, `pull_left_run.py`) over a shared
front-end (`grasp_scene.py`) → runtime builders (`choreographies.py`) → the
engine (`g1_choreograph.py`) → planning (`g1_plan_run.plan`) → hardware
(`g1_robot` + controllers). Perception is a separate Docker process feeding the
socket. Put-down/push and pull-left are in scope; walking is not.*