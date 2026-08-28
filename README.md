# g1_arm_motionplanning-rrt

Vision-guided **dual-arm pick-and-place** on the Unitree G1 humanoid — AprilTag
perception → RRT-Connect motion planning → dual-arm choreography, in pure Python
with **no ROS at runtime**.

The G1 uses a wrist-mounted RealSense D435i and two AprilTags (one on the box,
one on the table) to locate a box and perform a dual-arm side-pinch:

- **grab + lift** — approach → close in → grip → lift
- **put down + push** — place the box, release, and nudge it to the robot's right
- **pull left** — for a box jammed on the robot's right, drag it back toward the left

> Built in the WashU Humanoid Robotics Lab (Prof. Yiannis Kantaros). Custom
> RRT-Connect + damped-least-squares IK stack; a ROS 2 / MoveIt 2 prototype was
> dropped after hitting end-of-life ROS 2 Foxy constraints on the locomotion computer.

---

## Two entry points

```bash
python grab_run.py        # grab → lift → (optional) put-down + push right
python pull_left_run.py   # pull a far-right box back toward the left
```

## How it works

Two processes on the Jetson talk over a local TCP socket. `pyrealsense2` only
builds against Python 3.8, so perception runs in a Docker container while
planning/execution runs in the Python 3.12 conda env.

```
[Docker / Py3.8]  RealSense → AprilTag detect → solvePnP → T_optical_tag
        │  socket :5000
[conda / Py3.12]  perception_client → camera_calib → grasp_scene.snapshot_scene
                  → G1DualArm IK (14-DOF) → g1_plan_run.plan (RRT-Connect+shortcut)
                  → g1_choreograph.run_sequence → G1 arm/hand controllers
                                              │  unitree_sdk2py / CycloneDDS
                                              ▼  the robot moves
```

Every run snapshots the scene **once** (box + table poses, four grasp targets,
and two obstacle worlds — box-present and box-dropped), validates IK up front,
and only then plans and executes behind a confirm gate.

## Repo layout

| File | Role |
|---|---|
| `grab_run.py`, `pull_left_run.py` | **Entry points** — the two behaviors |
| `grasp_scene.py` | Shared front-end: tag snapshot → `Scene`, IK-validate, plan-HTML |
| `choreographies.py` | Runtime step builders: `build_grab` / `build_put_down` / `build_pull_left` |
| `g1_choreograph.py` | Choreography engine (`run_sequence`, `preflight_validate`) |
| `g1_plan_run.py` | Planning helpers: `plan` (RRT-Connect), `_validate`, obstacles, HTML |
| `rrt_planner.py`, `kdtree.py` | RRT-Connect + nearest-neighbor KD-tree |
| `myArm_v2.py`, `myArm_dual.py` | 7-DOF and 14-DOF kinematics, collision, analytical-Jacobian IK |
| `gravity_comp.py` | Pinocchio gravity feed-forward |
| `apriltag_pose_server.py` | Perception server (runs in the Docker container) |
| `perception_client.py`, `camera_calib.py`, `grasp_target.py` | Client, camera extrinsic, box/table geometry |
| `g1_robot.py`, `g1_arm_controller.py`, `g1_hand_controller.py`, `g1_arm_live_tuner.py` | Hardware layer |
| `g1_29dof_with_hand.urdf` | Robot model |
| **`DEVELOPMENT.md`** | **Full runbook, per-parameter tuning tables, calibration, troubleshooting** |

## Requirements

- **Hardware:** Unitree G1 (dual 7-DOF arms, dex3 hands), RealSense D435i, Jetson Orin NX.
- **Planning env (conda, Python 3.12):** `numpy`, `ikpy`, `pinocchio`, `unitree_sdk2py`, CycloneDDS.
- **Perception (Docker, Python 3.8):** `pyrealsense2`, `pupil-apriltags`, `opencv`.
  See `DEVELOPMENT.md` for the one-time image build.

## Quick start

```bash
# 0. Pre-position the right wrist (RWY) by hand — its motor is locked (see below).

# Terminal 1 — perception (Docker):
docker run -it --rm --network host --privileged -v /dev:/dev -v $(pwd):/work realsense:pyrs2 bash
python3 /work/apriltag_pose_server.py        # leave running

# Terminal 2 — the grab (conda):
conda activate base
python grab_run.py
```

## Notes

- **The right wrist-yaw motor is mechanically locked.** Its angle is read live at
  startup and pinned as a zero-width joint limit that IK/collision/planning honor.
  Pre-position it by hand before each run. Details in `DEVELOPMENT.md`.
- **Scope:** grab, put-down + push, and pull-left are implemented. The pull-left
  *drag* step is intentionally staged-but-disabled (post-drag hand extraction is
  still being designed). Walking is out of scope (that's the locomotion computer).
- **Tuning:** every knob (tag sizes, camera extrinsic, box/table geometry, grasp
  offsets, speeds, per-phase dwell) is documented with defaults in `DEVELOPMENT.md`.

## License

No license yet — add one before sharing publicly if you intend others to reuse it.
