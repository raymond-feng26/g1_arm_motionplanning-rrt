"""
camera_calib.py

Camera extrinsic calibration and frame-transform helpers.

Defines T_BASE_OPTICAL (the camera optical frame expressed in the robot's
base/torso frame) and the helper to map a tag pose from optical -> base.

Base frame convention:    x forward, y left, z up   (standard robot)
Optical frame convention: x right,   y down, z fwd  (OpenCV / solvePnP)

Mount geometry (per Unitree G1 docs / lab measurements):
    translation = (0.0476457, 0.0325, 0.4626817) m  in base frame
    pitch       = 42.4 deg downward  (about base +y axis)

VERIFY THE PITCH SIGN ON HARDWARE: hold a tag in front of the (pitched-down)
camera; T_base_tag should land BELOW the camera mount, not above. If z is
above the mount, flip CAMERA_PITCH_DEG sign and re-test.
"""
from typing import Tuple

import numpy as np


# --- mount parameters --------------------------------------------------------
CAMERA_MOUNT_XYZ = np.array([0.0476457, 0.0325, 0.3926817])  # base frame [m]
CAMERA_PITCH_DEG = 42.4   # downward; positive = camera nose tilts down


def _R_y(theta_rad: float) -> np.ndarray:
    """Rotation about base +y (left axis); positive = pitch down."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ])


# --- extrinsic, composed once ------------------------------------------------
# 1. Physical mount: pitched-down camera body, translated up the torso.
T_base_mount = np.eye(4)
T_base_mount[:3, :3] = _R_y(np.deg2rad(CAMERA_PITCH_DEG))
T_base_mount[:3, 3]  = CAMERA_MOUNT_XYZ

# 2. Optical->body axis swap (pure rotation, no translation):
#       x_body =  +z_opt    (forward  = optical forward)
#       y_body =  -x_opt    (left     = -optical right)
#       z_body =  -y_opt    (up       = -optical down)
T_mount_optical = np.eye(4)
T_mount_optical[:3, :3] = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
], dtype=np.float64)

# 3. The one constant the rest of the stack actually uses:
T_BASE_OPTICAL: np.ndarray = T_base_mount @ T_mount_optical


# --- helpers ----------------------------------------------------------------
def optical_to_base(T_optical_tag: np.ndarray) -> np.ndarray:
    """T_base_tag = T_base_optical @ T_optical_tag."""
    return T_BASE_OPTICAL @ T_optical_tag


def position_in_base(T_optical_tag: np.ndarray) -> np.ndarray:
    """Convenience: just the xyz translation in base frame."""
    return optical_to_base(T_optical_tag)[:3, 3]


# --- standalone sanity test -------------------------------------------------
def main():
    """Run the perception client and print T_base_tag.
    Hold the tag in front of the camera; verify base-frame xyz makes sense."""
    import time
    from perception_client import PerceptionClient

    np.set_printoptions(precision=4, suppress=True)
    print("T_BASE_OPTICAL =")
    print(T_BASE_OPTICAL)
    print()

    client = PerceptionClient(host="127.0.0.1", port=5000)
    client.start()
    print("[main] listening (Ctrl-C to stop)")
    last_t = 0.0
    try:
        while True:
            T_opt, t = client.latest_pose()
            if T_opt is not None and t != last_t:
                T_base = optical_to_base(T_opt)
                p_opt  = T_opt[:3, 3]
                p_base = T_base[:3, 3]
                print(f"\noptical: x={p_opt[0]:+.3f}  y={p_opt[1]:+.3f}  z={p_opt[2]:+.3f}")
                print(f"base:    x={p_base[0]:+.3f}  y={p_base[1]:+.3f}  z={p_base[2]:+.3f}  [m]")
                last_t = t
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[main] stopping")
    finally:
        client.stop()


if __name__ == "__main__":
    main()