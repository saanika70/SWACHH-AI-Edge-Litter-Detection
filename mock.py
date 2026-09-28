"""Hardware stand-in for laptops / CI / video-file testing."""
from __future__ import annotations

import logging
from typing import Any, Dict

from .base import Hardware, Presence

log = logging.getLogger("swachh.hw.mock")


class MockHardware(Hardware):
    def __init__(self, cfg: Dict[str, Any] | None = None):
        m = (cfg or {}).get("mock", {}) or {}
        self.always_present = bool(m.get("always_present", True))
        self.dark = bool(m.get("dark", True))
        self.light_level = 0.0
        self.events: list = []  # recorded actions (handy in tests)

    def read_presence(self) -> Presence:
        return Presence(pir=self.always_present, distance_m=None, present=self.always_present)

    def is_dark(self) -> bool:
        return self.dark

    def set_light(self, level: float) -> None:
        self.light_level = level
        self.events.append(("light", round(level, 2)))
        log.debug("LED -> %.2f", level)

    def strobe(self, duration_s: float) -> None:
        self.events.append(("strobe", duration_s))
        log.info("[mock] LED strobe %.1fs", duration_s)

    def beep(self, n: int, on_s: float, off_s: float) -> None:
        self.events.append(("beep", n))
        log.info("[mock] buzzer x%d", n)
