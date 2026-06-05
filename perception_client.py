"""
perception_client.py

Runs in your 3.12 conda env on humanoid. Connects to apriltag_pose_server.py
(in the realsense docker, port 5000), receives the live T_optical_tag (4x4,
OpenCV camera optical frame, tag bottom-left corner as the tag-frame origin),
and maintains the latest snapshot for any consumer to query.

Standalone test:
    python perception_client.py
    # should print the same matrices the server is printing
"""
import socket
import struct
import threading
import time
from typing import Optional, Tuple

import numpy as np

_MAT_SIZE   = 16 * 4            # 16 float32 = 64 bytes (one 4x4)
_TAG_SIZE   = 4 + _MAT_SIZE     # int32 id + matrix = 68 bytes
_COUNT_SIZE = 4                 # int32 tag-count prefix


class PerceptionClient:
    """Background-threaded TCP client. Keeps the latest T_optical_tag.

    Usage:
        client = PerceptionClient(host="127.0.0.1", port=5000)
        client.start()
        ...
        T, t = client.latest_pose()
        if T is not None:
            # use T (4x4 in camera optical frame), t is recv timestamp
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5000,
                 stale_after: float = 1.0):
        """
        host/port: where apriltag_pose_server is listening.
        stale_after: latest_pose() returns None if newest data older than this (s).
        """
        self.host = host
        self.port = port
        self.stale_after = stale_after
        # Latest snapshot per tag id: {tag_id: (T 4x4, recv_time)}.
        self._poses: dict = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def latest_pose(self, tag_id: int = 0
                    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Return (T_optical_tag, recv_time) for tag_id if fresh, else
        (None, None). Defaults to id 0 (the box) for backward compatibility."""
        with self._lock:
            entry = self._poses.get(tag_id)
            if entry is None:
                return None, None
            T, t_recv = entry
            if (time.time() - t_recv) > self.stale_after:
                return None, None
            return T.copy(), t_recv

    # ---- internal ----------------------------------------------------------

    def _connect(self) -> Optional[socket.socket]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect((self.host, self.port))
            print(f"[client] connected to {self.host}:{self.port}")
            return s
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            print(f"[client] connect failed ({e}); retrying in 1s...")
            return None

    def _recv_exact(self, s: socket.socket, n: int) -> Optional[bytes]:
        """Read exactly n bytes, checking stop flag periodically."""
        s.settimeout(0.5)
        buf = b""
        while len(buf) < n:
            if self._stop.is_set():
                return None
            try:
                chunk = s.recv(n - len(buf))
            except socket.timeout:
                continue        # no data yet; check stop and retry
            if not chunk:
                return None     # peer closed
            buf += chunk
        return buf

    def _loop(self):
        while not self._stop.is_set():
            s = self._connect()
            if s is None:
                time.sleep(1.0)
                continue
            try:
                while not self._stop.is_set():
                    hdr = self._recv_exact(s, _COUNT_SIZE)
                    if hdr is None:
                        print("[client] disconnected")
                        break
                    (count,) = struct.unpack("i", hdr)
                    now = time.time()
                    for _ in range(count):
                        body = self._recv_exact(s, _TAG_SIZE)
                        if body is None:
                            break
                        tag_id = struct.unpack("i", body[:4])[0]
                        floats = struct.unpack("16f", body[4:])
                        T = np.array(floats, dtype=np.float64).reshape(4, 4)
                        with self._lock:
                            self._poses[tag_id] = (T, now)
            finally:
                try: s.close()
                except Exception: pass


def main():
    np.set_printoptions(precision=4, suppress=True)
    client = PerceptionClient(host="127.0.0.1", port=5000)
    client.start()
    print("[main] listening for tag poses (Ctrl-C to stop)")
    try:
        while True:
            for tag_id in (0, 1):
                T, t = client.latest_pose(tag_id)
                if T is not None:
                    print(f"\n[id {tag_id}] T_optical_tag "
                          f"(recv {time.time() - t:.2f}s ago):")
                    print(T)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[main] stopping")
    finally:
        client.stop()


if __name__ == "__main__":
    main()