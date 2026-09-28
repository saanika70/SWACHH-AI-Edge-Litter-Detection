"""Overlay drawing and evidence-snapshot writing (with optional head blur)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .types import Box, LitterEvent

GREEN, RED, YELLOW, WHITE = (60, 200, 60), (40, 40, 230), (0, 200, 255), (255, 255, 255)


def _rect(img, box: Box, color, label: str = "", thick: int = 2) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    if label:
        cv2.putText(img, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def blur_heads(img: np.ndarray, person_boxes: Iterable[Box], head_frac: float = 0.3) -> np.ndarray:
    """Pixelate the top `head_frac` of every person box (privacy by design)."""
    h, w = img.shape[:2]
    for x1, y1, x2, y2 in person_boxes:
        x1, x2 = int(max(x1, 0)), int(min(x2, w))
        ya, yb = int(max(y1, 0)), int(min(y1 + head_frac * (y2 - y1), h))
        if x2 - x1 < 4 or yb - ya < 4:
            continue
        roi = img[ya:yb, x1:x2]
        small = cv2.resize(roi, (max((x2 - x1) // 12, 1), max((yb - ya) // 12, 1)))
        img[ya:yb, x1:x2] = cv2.resize(small, (x2 - x1, yb - ya), interpolation=cv2.INTER_NEAREST)
    return img


def draw_overlay(
    img: np.ndarray,
    persons: Sequence[Tuple[int, Box]],
    litters: Sequence[Tuple[int, str, Box]],
    status: str = "",
    watch_ids: Optional[set] = None,
) -> np.ndarray:
    out = img.copy()
    for pid, box in persons:
        color = YELLOW if watch_ids and pid in watch_ids else GREEN
        _rect(out, box, color, f"person #{pid}")
    for lid, label, box in litters:
        _rect(out, box, RED, f"{label} #{lid}")
    if status:
        cv2.putText(out, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2, cv2.LINE_AA)
    return out


class SnapshotWriter:
    def __init__(self, directory: str, max_files: int = 500, blur_people: bool = True):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_files = max_files
        self.blur_people = blur_people

    def save(
        self,
        img: np.ndarray,
        ev: LitterEvent,
        person_boxes: Sequence[Box],
        litter_boxes: Sequence[Tuple[str, Box]],
        device_id: str = "swachh",
    ) -> Path:
        frame = img.copy()
        if self.blur_people:
            blur_heads(frame, person_boxes)
        for b in person_boxes:
            _rect(frame, b, GREEN)
        for label, b in litter_boxes:
            _rect(frame, b, RED, label)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ev.wall_time))
        cv2.putText(frame, f"{ev.kind.upper()} {stamp}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2, cv2.LINE_AA)
        name = f"{device_id}_{time.strftime('%Y%m%d_%H%M%S', time.localtime(ev.wall_time))}_{ev.kind}.jpg"
        path = self.dir / name
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self._prune()
        return path

    def _prune(self) -> None:
        files: List[Path] = sorted(self.dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
        for old in files[: max(len(files) - self.max_files, 0)]:
            old.unlink(missing_ok=True)
