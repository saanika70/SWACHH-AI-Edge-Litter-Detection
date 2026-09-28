from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Presence:
    pir: bool
    distance_m: Optional[float]
    present: bool


class Hardware(ABC):
    """Interface implemented by the Raspberry Pi and mock back-ends."""

    @abstractmethod
    def read_presence(self) -> Presence: ...

    @abstractmethod
    def is_dark(self) -> bool: ...

    @abstractmethod
    def set_light(self, level: float) -> None:
        """Set LED brightness 0.0 - 1.0 (PWM)."""

    @abstractmethod
    def strobe(self, duration_s: float) -> None:
        """Flash the LED for `duration_s` seconds (non-blocking)."""

    @abstractmethod
    def beep(self, n: int, on_s: float, off_s: float) -> None:
        """Sound the buzzer `n` times (non-blocking)."""

    def battery(self) -> Optional[Dict[str, float]]:
        return None

    def close(self) -> None:
        pass
