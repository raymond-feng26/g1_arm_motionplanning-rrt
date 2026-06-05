"""
apriltag_pose_server.py

Runs in the realsense docker (Python 3.8 + pyrealsense2 + pupil_apriltags + cv2).
Grabs RealSense color frames, detects an AprilTag, solves PnP, and:
  - prints the 4x4 T_optical_tag every detection (so you can run it standalone)
  - serves the latest 4x4 over TCP to any connected client (16 float32 = 64 bytes)

T_optical_tag is in the OpenCV camera optical frame:
    x = right, y = down, z = forward (into scene)
Tag frame origin = tag bottom-left corner; +x along bottom edge,
+y up the left edge, +z out of the tag plane.

The client side (your 3.12 conda stack) applies the camera-to-base extrinsic.
"""

import socket
import struct
import threading

import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector


# --- Config -----------------------------------------------------------------
# Physical side length per tag ID [m]. Detection is family-wide, so all IDs of
# TAG_FAMILY are found in one pass; only PnP needs the size, so a per-ID map
# lets differently-sized tags coexist (box tag is small, table tag is larger).
TAG_SIZES      = {0: 0.034,     # box tag
                  1: 0.08}      # table tag
RES_W, RES_H   = 640, 480       # color stream resolution
FPS            = 6
SOCKET_PORT    = 5000
TAG_FAMILY     = "tag36h11"
# wait_for_frames timeout [ms]. librealsense defaults to 5000 and RAISES on
# expiry, which would kill the server on a single USB hiccup. We catch the
# timeout in the loop and retry/restart instead, but also give it more slack.
FRAME_TIMEOUT_MS = 10000
# ----------------------------------------------------------------------------


def make_obj_points(tag_len):
    """Tag-frame corner coordinates. Origin = bottom-left corner.
    Order matches pupil_apriltags' corner order (CCW from BL: BL, BR, TR, TL)."""
    return np.array([
        [0,        0,        0],   # bottom-left
        [tag_len,  0,        0],   # bottom-right
        [tag_len,  tag_len,  0],   # top-right
        [0,        tag_len,  0],   # top-left
    ], dtype=np.float64)


def start_camera():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, RES_W, RES_H, rs.format.bgr8, FPS)
    profile = pipeline.start(cfg)

    stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = stream.get_intrinsics()
    K = np.array([[intr.fx, 0,       intr.ppx],
                  [0,       intr.fy, intr.ppy],
                  [0,       0,       1.0     ]], dtype=np.float64)
    D = np.array(intr.coeffs, dtype=np.float64)
    print(f"[camera] {RES_W}x{RES_H}@{FPS} "
          f"fx={intr.fx:.2f} fy={intr.fy:.2f} "
          f"ppx={intr.ppx:.2f} ppy={intr.ppy:.2f}"
    )
    print(f"[camera] distortion = {D}")
    return pipeline, K, D


def restart_camera(pipeline):
    """Stop and re-start the pipeline after a frame stall. Returns a fresh
    (pipeline, K, D). Used when wait_for_frames keeps timing out."""
    try:
        pipeline.stop()
    except Exception as e:
        print(f"[camera] stop during restart raised (ignored): {e}")
    print("[camera] restarting pipeline...")
    return start_camera()


class PoseServer:
    """TCP server that pushes a 4x4 (16 float32) to a connected client.
    Accept runs in a daemon thread; main loop just calls send()."""

    def __init__(self, port):
        self.port = port
        self.client = None
        self.lock = threading.Lock()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', port))
        self.sock.listen(1)
        threading.Thread(target=self._accept_loop, daemon=True).start()
        print(f"[socket] listening on 0.0.0.0:{port}")

    def _accept_loop(self):
        while True:
            client, addr = self.sock.accept()
            print(f"[socket] client connected from {addr}")
            with self.lock:
                if self.client is not None:
                    try: self.client.close()
                    except Exception: pass
                self.client = client

    def send_poses(self, poses):
        """Send one frame of detected poses over TCP:
            int32 count, then per tag: int32 id + 16 float32 (row-major 4x4).
        poses = list of (tag_id, T). Sent every frame (count may be 0) so the
        client stays live and can age out tags that dropped from view."""
        with self.lock:
            if self.client is None:
                return
            buf = struct.pack('i', len(poses))
            for tag_id, T in poses:
                buf += struct.pack('i', int(tag_id))
                buf += struct.pack('16f', *T.flatten().astype(np.float32))
            try:
                self.client.sendall(buf)
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                print(f"[socket] client dropped: {e}")
                try: self.client.close()
                except Exception: pass
                self.client = None


def main():
    np.set_printoptions(precision=4, suppress=True)

    pipeline, K, D = start_camera()
    detector = Detector(families=TAG_FAMILY)       # built ONCE, not per frame
    # Per-ID object points — corner coords scale with each tag's physical size.
    obj_points = {tid: make_obj_points(size) for tid, size in TAG_SIZES.items()}
    server = PoseServer(SOCKET_PORT)

    try:
        consecutive_timeouts = 0
        while True:
            # wait_for_frames RAISES on timeout. Catch it so one USB hiccup
            # doesn't kill the server; after a few in a row, restart the
            # pipeline (handles autosuspend / transient USB drops).
            try:
                frames = pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
            except RuntimeError as e:
                consecutive_timeouts += 1
                print(f"\n[camera] wait_for_frames timeout "
                      f"#{consecutive_timeouts} ({e})")
                if consecutive_timeouts >= 3:
                    pipeline, K, D = restart_camera(pipeline)
                    consecutive_timeouts = 0
                continue
            consecutive_timeouts = 0

            color = frames.get_color_frame()
            if not color:
                continue
            img  = np.asanyarray(color.get_data())          # H x W x 3, BGR
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            tags = detector.detect(gray)

            # Solve PnP for every wanted tag in view, each with its own size.
            # Tags whose IDs aren't in TAG_SIZES are ignored.
            poses = []
            for tag in tags:
                if tag.tag_id not in obj_points:
                    continue
                img_points = np.asarray(tag.corners, dtype=np.float64)
                ok, rvec, tvec = cv2.solvePnP(obj_points[tag.tag_id],
                                              img_points, K, D)
                if not ok:
                    print(f"[pnp] solvePnP failed for id {tag.tag_id}")
                    continue
                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3]  = tvec.flatten()
                poses.append((tag.tag_id, T))

            if poses:
                print(f"[tags {','.join(str(i) for i, _ in poses)}] "
                      f"sent {len(poses)}")
            else:
                print(".", end="", flush=True)   # nothing wanted in view

            server.send_poses(poses)             # heartbeat even when empty

    except KeyboardInterrupt:
        print("\n[main] interrupted")
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        print("[main] pipeline stopped")


if __name__ == "__main__":
    main()