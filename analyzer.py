"""Glue: person tracker + litter tracker + event engine."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from .event_engine import AnalysisConfig, EngineResult, LitterEventEngine
from .tracker import Tracker
from .types import Detection


class LitterAnalyzer:
    def __init__(self, analysis_cfg: Optional[Dict[str, Any]] = None):
        a = dict(analysis_cfg or {})
        trk = a.get("tracking", {}) or {}
        self.person_tracker = Tracker(
            max_dist_px=trk.get("person_max_dist_px", 150),
            max_age_s=trk.get("person_max_age_s", 1.5),
        )
        self.litter_tracker = Tracker(
            max_dist_px=trk.get("litter_max_dist_px", 120),
            max_age_s=trk.get("litter_max_age_s", 1.0),
        )
        self.engine = LitterEventEngine(AnalysisConfig.from_dict(a))

    def step(
        self,
        t: float,
        person_dets: Optional[Sequence[Detection]],
        litter_dets: Sequence[Detection],
        frame_size: Tuple[int, int],
    ) -> EngineResult:
        """`person_dets=None` means "person detector skipped this frame"."""
        if person_dets is not None:
            self.person_tracker.update(person_dets, t)
        self.litter_tracker.update(litter_dets, t)
        return self.engine.process(
            t,
            list(self.person_tracker.tracks.values()),
            list(self.litter_tracker.tracks.values()),
            frame_size,
        )
