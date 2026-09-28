"""Minimal multi-object tracker (greedy, velocity-predicted, label-aware).

Deliberately dependency-free so it runs comfortably on a Raspberry Pi and can
be unit-tested without OpenCV/PyTorch.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Sequence, Tuple

from .types import Box, Detection


class Track:
    def __init__(self, track_id: int, det: Detection, t: float, history_len: int = 40):
        self.id = track_id
        self.label = det.label
        self.box: Box = det.box
        self.conf = det.conf
        self.first_seen = t
        self.last_seen = t
        self.hits = 1
        self.updated = True
        cx, cy = det.center
        self.history: deque = deque([(t, cx, cy)], maxlen=history_len)

    # -- geometry ---------------------------------------------------------
    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def cx(self) -> float:
        return self.center[0]

    @property
    def cy(self) -> float:
        return self.center[1]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    # -- motion -----------------------------------------------------------
    def velocity(self, window_s: float = 0.4) -> Tuple[float, float]:
        """Average velocity (px/s) over the most recent `window_s` seconds."""
        pts = [p for p in self.history if self.last_seen - p[0] <= window_s]
        if len(pts) < 2:
            return 0.0, 0.0
        (t0, x0, y0), (t1, x1, y1) = pts[0], pts[-1]
        dt = t1 - t0
        if dt <= 1e-6:
            return 0.0, 0.0
        return (x1 - x0) / dt, (y1 - y0) / dt

    def predict_center(self, t: float) -> Tuple[float, float]:
        vx, vy = self.velocity()
        dt = min(max(t - self.last_seen, 0.0), 1.0)
        cx, cy = self.center
        return cx + vx * dt, cy + vy * dt

    def box_at(self, t: float) -> Box:
        """Box extrapolated to time `t` (useful when a person was not re-detected)."""
        vx, vy = self.velocity()
        dt = min(max(t - self.last_seen, 0.0), 1.0)
        dx, dy = vx * dt, vy * dt
        x1, y1, x2, y2 = self.box
        return x1 + dx, y1 + dy, x2 + dx, y2 + dy

    def update(self, det: Detection, t: float) -> None:
        self.box = det.box
        self.conf = det.conf
        self.last_seen = t
        self.hits += 1
        self.updated = True
        cx, cy = det.center
        self.history.append((t, cx, cy))


class Tracker:
    def __init__(self, max_dist_px: float = 120.0, max_age_s: float = 1.0, history_len: int = 40):
        self.max_dist = max_dist_px
        self.max_age_s = max_age_s
        self.history_len = history_len
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: Sequence[Detection], t: float) -> List[Track]:
        for tr in self.tracks.values():
            tr.updated = False

        pairs = []
        for tid, tr in self.tracks.items():
            px, py = tr.predict_center(t)
            for di, det in enumerate(detections):
                if det.label != tr.label:
                    continue
                cx, cy = det.center
                dist = math.hypot(cx - px, cy - py)
                if dist <= self.max_dist:
                    pairs.append((dist, tid, di))
        pairs.sort()

        used_tracks, used_dets = set(), set()
        for _, tid, di in pairs:
            if tid in used_tracks or di in used_dets:
                continue
            self.tracks[tid].update(detections[di], t)
            used_tracks.add(tid)
            used_dets.add(di)

        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            self.tracks[self._next_id] = Track(self._next_id, det, t, self.history_len)
            self._next_id += 1

        for tid in [k for k, tr in self.tracks.items() if t - tr.last_seen > self.max_age_s]:
            del self.tracks[tid]

        return list(self.tracks.values())
