# Hardware guide

## Bill of materials (reference build)

| Part | Notes |
|---|---|
| Raspberry Pi 4 (4 GB) or Pi 5 | Pi 3 works only at very low FPS |
| Camera | Pi Camera Module 3 (CSI) or a USB webcam |
| PIR sensor HC-SR501 | wake-up trigger |
| Ultrasonic sensor HC-SR04 | presence / distance (needs level shifting on ECHO) |
| LDR module (digital out) | day / night detection |
| LED street-light module, 12 V, ~10 W | driven through a logic-level N-MOSFET (e.g. IRLZ44N) |
| Active buzzer (5 V) | driven through an NPN transistor (e.g. 2N2222) |
| INA219 breakout *(optional)* | battery voltage/current telemetry |
| Solar panel + MPPT charge controller + 12 V battery | see sizing below |
| 12 V → 5.1 V / 3 A buck converter | quality converter for the Pi |
| Weatherproof enclosure, fuses, wiring | outdoor use |

## Wiring (BCM pin numbers — configurable in `config/config.yaml`)

| Signal | Pi pin (BCM) | Notes |
|---|---|---|
| PIR OUT | GPIO 17 | 5 V supply; OUT is 3.3 V-safe on HC-SR501 |
| Ultrasonic TRIG | GPIO 23 | direct |
| Ultrasonic ECHO | GPIO 24 | **voltage divider 1 kΩ / 2 kΩ** (5 V → 3.3 V) |
| LED PWM | GPIO 18 | → MOSFET gate via 100 Ω, 10 kΩ pull-down; LED on the 12 V load rail |
| Buzzer | GPIO 27 | → transistor base via 1 kΩ |
| LDR DO | GPIO 22 | flip `ldr_dark_is_high` if your module is inverted |
| INA219 SDA / SCL | GPIO 2 / 3 | I²C, address 0x40 |

```
12V battery ─┬─ buck 5.1V ── Pi 5V
             ├─ INA219 (high-side shunt) ── load
             └─ LED module ── MOSFET drain ; source ── GND ; gate ── GPIO18 (PWM)
```

> Common ground between the Pi, MOSFET/transistor stages and the battery is essential.

## Solar sizing (illustrative — measure your own with the INA219)

Daily energy = Σ (power × hours):

| Load | Assumption | Energy |
|---|---|---|
| Raspberry Pi + camera | ~4 W average (sensor-gated inference) × 24 h | 96 Wh |
| LED idle at night | 10 W × 15 % × 12 h | 18 Wh |
| LED active | 10 W × 100 % × 1 h | 10 Wh |
| Sensors, losses | — | ~25 Wh |
| **Total** | | **≈ 150 Wh/day** |

- **Panel** ≈ daily energy ÷ (peak-sun-hours × system efficiency) = 150 ÷ (5 × 0.7) ≈ **43 W → use a 50 W panel** (adjust for your location and season).
- **Battery** for 2 days autonomy: 300 Wh ÷ 0.8 usable ≈ 375 Wh → **12.8 V × 30 Ah LiFePO₄**.
- Sensor-gated inference is the biggest lever: raising idle time or lowering `active_fps` directly shrinks the panel and battery you need.

## Software notes

- Pi OS **Bookworm 64-bit**; run `scripts/install_pi.sh` (installs `python3-lgpio` for Pi 5 GPIO and `python3-picamera2` for CSI cameras).
- For a CSI camera set `camera.backend: picamera2`.
- Export the model to **NCNN** and benchmark with `scripts/benchmark.py`; choose `imgsz` / `active_fps` from the measured speed.
- Add a heatsink/fan — sustained inference throttles a bare Pi in an enclosure.
- Test outputs first with `--mock-hw`, then with real GPIO before mounting the unit on a pole. Treat 12 V wiring and the enclosure/mounting with proper electrical safety.
