"""
grasp_target.py

Box-on-table grasping geometry. Consumes the live tag pose, applies the
tag-on-box geometry (T_box_tag), and computes the box center plus the
pre-grasp / grasp positions for the two-handed side pinch.

Current setup
-------------
Box:  28 cm (x, depth, fwd-back) x 30 cm (y, width, left-right) x 12 cm (z, height)
Tag:  35 mm tag36h11, on TOP face. Bottom-LEFT corner of tag = center of top face.
      Tag orientation on box (ASSUMED — verify physically and adjust R_box_tag):
        tag +y -> box +x   (tag "up" points away from robot)
        tag +x -> box -y   (tag "right" points to robot's right side)
        tag +z -> box +z   (out of top face, up)

Grasp strategy
--------------
Two-handed side pinch. Hands close along box's y-axis (the 30 cm dimension).
  pre-grasp:  ee at box center +/- (half_box_y + clearance)  along y
  grasp:      ee at box center +/- half_box_y                along y
Position only for now; approach orientation comes next.
"""
import time

import numpy as np

from camera_calib import optical_to_base
from perception_client import PerceptionClient


# --- box geometry (meters) --------------------------------------------------
BOX_X_LEN = 0.28
BOX_Y_LEN = 0.30
BOX_Z_LEN = 0.12


# --- tag-on-box: tag pose in box frame --------------------------------------
# Columns of R_box_tag = tag's axes expressed in box frame:
#   tag +x in box = (0, -1, 0)   (i.e., tag +x  = -box +y, "robot's right")
#   tag +y in box = (1,  0, 0)   (i.e., tag +y  = +box +x, "away from robot")
#   tag +z in box = (0,  0, 1)   (i.e., tag +z  = +box +z, "up")
R_box_tag = np.array([
    [ 0.0,  1.0,  0.0],
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0],
])
T_box_tag = np.eye(4)
T_box_tag[:3, :3] = R_box_tag
T_box_tag[:3, 3]  = np.array([0.0, 0.0, BOX_Z_LEN / 2.0])   # tag corner sits on top face

# Inverse: box pose in tag frame, used to chain T_base_box = T_base_tag @ T_tag_box.
def _invert_rigid(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    inv = np.eye(4)
    inv[:3, :3] = R.T
    inv[:3, 3]  = -R.T @ t
    return inv

T_tag_box = _invert_rigid(T_box_tag)


# --- helpers ----------------------------------------------------------------
def box_in_base(T_base_tag):
    """T_base_box = T_base_tag @ T_tag_box, with pitch/roll zeroed.

    Box is on a flat table — keep only yaw. Eliminates AABB inflation
    from noisy tag-normal direction, and equalizes left/right grasp z.
    """
    T = T_base_tag @ T_tag_box

    yaw = np.arctan2(T[1, 0], T[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    T_flat = np.eye(4)
    T_flat[:3, :3] = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])
    T_flat[:3, 3] = T[:3, 3]
    return T_flat



# --- table geometry (meters) ------------------------------------------------
# Table located by its own AprilTag (id 1, 8 cm) lying flat on the table top
# at the middle of the FAR side. Mounting convention ASSUMED the same as the
# box top tag (verify physically; flip a sign in R_table_tag if wrong):
#   tag +x -> table -y   (tag "right" = robot's right)
#   tag +y -> table +x   (tag "up" points toward the far edge, away from robot)
#   tag +z -> table +z   (out of the table top, up)
# Tag bottom-left corner = table-frame origin, sitting ON the table top.
TABLE_TAG_LEN    = 0.08
TABLE_THICK      = 0.04
TABLE_WIDTH      = 2.0       # "very wide" — y-extent is not located from the tag
TAG_TO_FAR_EDGE  = 0.09      # tag BL corner -> far edge,  along table +x
TAG_TO_NEAR_EDGE = 0.517     # tag BL corner -> near edge, along table +x (toward robot)
TABLE_DEPTH      = TAG_TO_FAR_EDGE + TAG_TO_NEAR_EDGE      # 0.607 m, close -> far

R_table_tag = np.array([
    [ 0.0,  1.0,  0.0],
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0],
])
T_table_tag = np.eye(4)
T_table_tag[:3, :3] = R_table_tag
# table-frame origin = tag BL corner on the table top -> zero translation.
T_tag_table = _invert_rigid(T_table_tag)


def table_in_base(T_base_tag1):
    """T_base_table from the table tag (id 1), pitch/roll zeroed (yaw only).

    Table frame: origin = tag bottom-left corner (on the table top),
    +x = depth toward the far edge, +y = width, +z = up. Mirrors box_in_base:
    the table is horizontal, so keep only yaw and drop tag-normal noise.
    """
    T = T_base_tag1 @ T_tag_table
    yaw = np.arctan2(T[1, 0], T[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    T_flat = np.eye(4)
    T_flat[:3, :3] = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])
    T_flat[:3, 3] = T[:3, 3]
    return T_flat


def grasp_targets(T_base_box, clearance=0.15, pregrasp_z_lift=0.0):
    # ---- 自定义调整(都是 box 坐标系) ----
    GRASP_X_OFFSET = -0.09      # 沿 box 深度,0 = 中间。正 = 远离机器人
    GRASP_Z_OFFSET = -0.00      # 沿 box 高度,正 = 向上(往 top 靠近)。这里设 4cm,贴近 top edge 利于上抬
    HAND_PENETRATION = -0.01    # 指尖比 box 侧面深入多少米;0 = 贴外表,正 = 进去
    # pregrasp_z_lift: pre-grasp sits this much HIGHER than the grasp, so the
    # close-in step descends as it closes (box +z == base +z, since box_in_base
    # zeroes pitch/roll, so this is a clean vertical lift regardless of yaw).
    # ---------------------------------------------

    half = BOX_Y_LEN / 2.0 - HAND_PENETRATION

    def at_box(y_off, z_extra=0.0):
        p = np.array([GRASP_X_OFFSET, y_off, GRASP_Z_OFFSET + z_extra, 1.0])
        return (T_base_box @ p)[:3]

    return {
        'box_center':  at_box(0.0),
        'left_pre':    at_box(+(half + clearance), pregrasp_z_lift),
        'right_pre':   at_box(-(half + clearance), pregrasp_z_lift),
        'left_grasp':  at_box(+half),
        'right_grasp': at_box(-half),
    }


# --- standalone visualization ----------------------------------------------
def _fmt(p):
    return f"({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})"


def main():
    np.set_printoptions(precision=4, suppress=True)
    client = PerceptionClient(host="127.0.0.1", port=5000)
    client.start()
    print("[main] listening (Ctrl-C to stop)")
    last_t = 0.0
    try:
        while True:
            T_opt, t = client.latest_pose()
            if T_opt is not None and t != last_t:
                T_base_tag = optical_to_base(T_opt)
                T_base_box = box_in_base(T_base_tag)
                g = grasp_targets(T_base_box)

                # box yaw in base frame (rotation about z) for orientation eyeball
                yaw_deg = np.degrees(np.arctan2(T_base_box[1, 0], T_base_box[0, 0]))

                print(f"\nbox_center {_fmt(g['box_center'])}  yaw={yaw_deg:+.1f} deg")
                print(f"  left_pre   {_fmt(g['left_pre'])}")
                print(f"  left_grasp {_fmt(g['left_grasp'])}")
                print(f"  right_pre  {_fmt(g['right_pre'])}")
                print(f"  right_grasp{_fmt(g['right_grasp'])}")
                last_t = t
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[main] stopping")
    finally:
        client.stop()


if __name__ == "__main__":
    main()