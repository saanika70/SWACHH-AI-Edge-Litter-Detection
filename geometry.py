"""Small geometry helpers used by the person-litter proximity analysis."""
from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

from .types import Box


def box_center(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_to_box_distance(px: float, py: float, box: Box) -> float:
    """Euclidean distance from a point to a box (0 if the point is inside)."""
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def proximity_ratio(litter_box: Box, person_box: Box) -> float:
    """Distance from the litter centre to the person box, in *person heights*.

    Normalising by the person's height makes thresholds independent of camera
    distance/zoom: 0.0 means the litter centre is inside the person box, 1.0
    means it is one body-height away from the person.
    """
    cx, cy = box_center(litter_box)
    height = max(person_box[3] - person_box[1], 1.0)
    return point_to_box_distance(cx, cy, person_box) / height


def in_zones(
    center: Tuple[float, float],
    zones: Iterable[Sequence[float]],
    frame_w: float,
    frame_h: float,
) -> bool:
    """True if `center` falls inside any normalised [x1, y1, x2, y2] zone."""
    cx, cy = center
    for x1, y1, x2, y2 in zones:
        if x1 * frame_w <= cx <= x2 * frame_w and y1 * frame_h <= cy <= y2 * frame_h:
            return True
    return False
