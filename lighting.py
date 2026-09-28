"""Adaptive LED lighting: dim when idle, bright when someone is present,
brightest / strobing when littering is detected."""
from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, Optional

from .base import Hardware


class Mode(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    WATCH = "watch"
    ALERT = "alert"


DEFAULT_LEVELS = {
    "idle": {"day": 0.0, "night": 0.15},
    "active": {"day": 0.0, "night": 0.70},
    "watch": {"day": 0.5, "night": 1.0},
    "alert": {"day": 1.0, "night": 1.0},
}


class LightingController:
    def __init__(self, hw: Hardware, cfg: Optional[Dict[str, Any]] = None):
        c = cfg or {}
        self.hw = hw
        self.levels = {**DEFAULT_LEVELS, **(c.get("levels") or {})}
        self.fade_per_s = float(c.get("fade_per_s", 1.5))
        self.strobe_s = float(c.get("alert_strobe_s", 6.0))
        self._level = 0.0
        self._applied: Optional[float] = None
        self._last_t = time.monotonic()
        self._strobe_until = 0.0

    def target(self, mode: Mode, dark: bool) -> float:
        return float(self.levels[Mode(mode).value]["night" if dark else "day"])

    def update(self, mode: Mode, dark: bool) -> None:
        now = time.monotonic()
        dt, self._last_t = now - self._last_t, now
        if now < self._strobe_until:
            return  # let the strobe run
        if self._strobe_until:
            self._strobe_until, self._applied = 0.0, None  # strobe finished: re-apply level
        tgt = self.target(mode, dark)
        step = self.fade_per_s * dt
        self._level = min(tgt, self._level + step) if tgt > self._level else max(tgt, self._level - step)
        if self._applied is None or abs(self._level - self._applied) >= 0.01:
            self.hw.set_light(self._level)
            self._applied = self._level

    def alert(self) -> None:
        self._strobe_until = time.monotonic() + self.strobe_s
        self._level = 1.0
        self.hw.strobe(self.strobe_s)
