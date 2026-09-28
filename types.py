"""Lightweight data types shared across the package (no heavy imports)."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass(frozen=True)
class Detection:
    label: str
    conf: float
    box: Box

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


@dataclass
class LitterEvent:
    """A littering-related event produced by the analysis engine."""

    kind: str  # "throw" | "drop" | "accumulation"
    timestamp: float  # analysis clock (video time or monotonic seconds)
    wall_time: float = field(default_factory=time.time)
    person_id: Optional[int] = None
    litter_id: Optional[int] = None
    label: str = ""
    confidence: float = 0.0
    litter_box: Optional[Box] = None
    person_box: Optional[Box] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
