"""Automated waste monitoring: flag areas where litter keeps piling up."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .geometry import in_zones, proximity_ratio
from .types import Detection, LitterEvent


class AccumulationMonitor:
    """Counts litter lying on the ground and raises an event when the count
    stays at/above ``threshold`` for ``consecutive_scans`` scans in a row.

    Litter next to a person (probably being held) and litter inside exclusion
    zones (e.g. a dustbin) are ignored.
    """

    def __init__(
        self,
        threshold: int = 5,
        consecutive_scans: int = 2,
        cooldown_s: float = 1800.0,
        near_ratio: float = 0.6,
        exclusion_zones: Optional[list] = None,
    ):
        self.threshold = threshold
        self.consecutive_scans = consecutive_scans
        self.cooldown_s = cooldown_s
        self.near_ratio = near_ratio
        self.zones = exclusion_zones or []
        self.streak = 0
        self.last_count = 0
        self._last_emit: Optional[float] = None

    def count_ground_litter(
        self,
        litters: Sequence[Detection],
        persons: Sequence[Detection],
        frame_size: Tuple[int, int],
    ) -> int:
        fw, fh = frame_size
        n = 0
        for d in litters:
            if in_zones(d.center, self.zones, fw, fh):
                continue
            if any(proximity_ratio(d.box, p.box) <= self.near_ratio for p in persons):
                continue
            n += 1
        return n

    def update(
        self,
        t: float,
        litters: Sequence[Detection],
        persons: Sequence[Detection],
        frame_size: Tuple[int, int],
    ) -> Optional[LitterEvent]:
        self.last_count = self.count_ground_litter(litters, persons, frame_size)
        self.streak = self.streak + 1 if self.last_count >= self.threshold else 0
        if self.streak < self.consecutive_scans:
            return None
        if self._last_emit is not None and (t - self._last_emit) < self.cooldown_s:
            return None
        self._last_emit = t
        return LitterEvent(
            kind="accumulation",
            timestamp=t,
            label="litter_pile",
            extra={"count": self.last_count, "threshold": self.threshold},
        )
