"""Raspberry Pi GPIO back-end built on gpiozero (Pi 3/4/5, Pi OS Bookworm).

Wiring is documented in docs/hardware.md.  Pins are configurable in
config/config.yaml (BCM numbering).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .base import Hardware, Presence

log = logging.getLogger("swachh.hw.pi")


class RaspberryPiHardware(Hardware):
    def __init__(self, cfg: Dict[str, Any]):
        from gpiozero import Buzzer, DigitalInputDevice, DistanceSensor, MotionSensor, PWMLED

        pins = cfg.get("pins", {})
        self.presence_distance_m = float(cfg.get("presence_distance_m", 3.0))
        self.dark_is_high = bool(cfg.get("ldr_dark_is_high", True))

        self.pir = MotionSensor(pins.get("pir", 17))
        self.ultra = DistanceSensor(
            echo=pins.get("ultrasonic_echo", 24),
            trigger=pins.get("ultrasonic_trigger", 23),
            max_distance=float(cfg.get("ultrasonic_max_m", 4.0)),
        )
        self.led = PWMLED(pins.get("led_pwm", 18))
        self.buzzer = Buzzer(pins.get("buzzer", 27))
        self.ldr = DigitalInputDevice(pins.get("ldr", 22), pull_up=False)

        self._ina = None
        bat = cfg.get("battery", {}) or {}
        if bat.get("enabled"):
            try:
                from ina219 import INA219  # pip install pi-ina219

                self._ina = INA219(
                    float(bat.get("shunt_ohms", 0.1)),
                    float(bat.get("max_amps", 3.2)),
                    address=int(bat.get("i2c_address", 0x40)),
                )
                self._ina.configure()
            except Exception as exc:  # pragma: no cover - hardware specific
                log.warning("INA219 unavailable: %s", exc)
        log.info("Raspberry Pi GPIO initialised")

    def read_presence(self) -> Presence:
        pir = bool(self.pir.motion_detected)
        dist: Optional[float] = None
        try:
            dist = float(self.ultra.distance)
        except Exception:  # noisy echo
            pass
        near = dist is not None and dist < min(self.presence_distance_m, self.ultra.max_distance * 0.98)
        return Presence(pir=pir, distance_m=dist, present=pir or near)

    def is_dark(self) -> bool:
        v = bool(self.ldr.value)
        return v if self.dark_is_high else not v

    def set_light(self, level: float) -> None:
        self.led.value = max(0.0, min(1.0, level))

    def strobe(self, duration_s: float) -> None:
        n = max(int(duration_s / 0.24), 1)
        self.led.blink(on_time=0.12, off_time=0.12, n=n, background=True)

    def beep(self, n: int, on_s: float, off_s: float) -> None:
        self.buzzer.beep(on_time=on_s, off_time=off_s, n=n, background=True)

    def battery(self) -> Optional[Dict[str, float]]:
        if self._ina is None:
            return None
        try:
            return {"bus_v": round(self._ina.voltage(), 2), "current_ma": round(self._ina.current(), 1)}
        except Exception:  # pragma: no cover
            return None

    def close(self) -> None:
        for dev in (self.pir, self.ultra, self.led, self.buzzer, self.ldr):
            try:
                dev.close()
            except Exception:
                pass
