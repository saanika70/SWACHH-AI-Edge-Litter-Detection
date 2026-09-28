"""Camera abstraction: OpenCV (USB/RTSP/video file) or Picamera2 (CSI)."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

log = logging.getLogger("swachh.camera")


@dataclass
class Frame:
    image: np.ndarray
    ts: float  # seconds: video time for files, monotonic clock for live sources
    index: int


class Camera:
    def __init__(self, cfg: Dict[str, Any]):
        self.backend = cfg.get("backend", "opencv")
        self.source = cfg.get("source", 0)
        self.width = int(cfg.get("width", 640))
        self.height = int(cfg.get("height", 480))
        self.fps = int(cfg.get("fps", 30))
        self.is_file = False
        self._cap = None
        self._picam = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._new = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._seq = 0
        self._file_fps = 30.0

    def open(self) -> None:
        if self.backend == "picamera2":
            from picamera2 import Picamera2

            self._picam = Picamera2()
            conf = self._picam.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}  # BGR order in memory
            )
            self._picam.configure(conf)
            self._picam.start()
            log.info("Picamera2 started at %dx%d", self.width, self.height)
            return

        import cv2

        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {src!r}")
        self.is_file = isinstance(src, str) and Path(src).is_file()
        if self.is_file:
            self._file_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            log.info("Video file %s @ %.1f fps", src, self._file_fps)
        else:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
            log.info("Live camera %r started", src)

    def _reader(self) -> None:
        while not self._stop.is_set():
            ok, img = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._seq += 1
                self._latest = (img, time.monotonic(), self._seq)
            self._new.set()

    def read(self, timeout: float = 1.0) -> Optional[Frame]:
        if self._picam is not None:
            img = self._picam.capture_array()
            self._seq += 1
            return Frame(img, time.monotonic(), self._seq)
        if self.is_file:
            ok, img = self._cap.read()
            if not ok:
                return None
            self._seq += 1
            return Frame(img, (self._seq - 1) / self._file_fps, self._seq)
        if not self._new.wait(timeout):
            return None
        with self._lock:
            img, ts, seq = self._latest
            self._new.clear()
        return Frame(img, ts, seq)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
        if self._picam is not None:
            self._picam.stop()
            self._picam.close()
