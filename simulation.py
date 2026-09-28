"""Synthetic scenarios so the analysis logic can be tested/demoed with no
camera, model or hardware.  Each scenario yields ``(t, persons, litters)``."""
from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np

from .analyzer import LitterAnalyzer
from .types import Detection, LitterEvent

FRAME_SIZE = (640, 480)
FPS = 10
PERSON_W, PERSON_H = 110, 300

Frame = Tuple[float, List[Detection], List[Detection]]


def _box(cx: float, cy: float, w: float, h: float):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _person(rng, cx, cy) -> Detection:
    j = rng.normal(0, 1.0, 2)
    return Detection("person", 0.9, _box(cx + j[0], cy + j[1], PERSON_W, PERSON_H))


def _bottle(rng, cx, cy) -> Detection:
    j = rng.normal(0, 1.0, 2)
    return Detection("plastic_bottle", 0.8, _box(cx + j[0], cy + j[1], 28, 44))


def scenario_throw(seed: int = 0, fps: int = FPS, duration_s: float = 6.0) -> Iterator[Frame]:
    """A person walks slowly holding a bottle, then hurls it away."""
    rng = np.random.default_rng(seed)
    pos = vel = None
    for i in range(int(duration_s * fps)):
        t = i / fps
        pcx, pcy = 100 + 20 * t, 300
        hand = (pcx + 45, pcy - 20)
        if t < 2.0:
            bx, by = hand
        else:
            if pos is None:
                pos, vel = list(hand), [520.0, -160.0]
            vel[1] += 600.0 / fps  # gravity
            pos[0] += vel[0] / fps
            pos[1] += vel[1] / fps
            if pos[1] >= 420:
                pos[1], vel = 420, [0.0, 0.0]
            bx, by = pos
        yield t, [_person(rng, pcx, pcy)], [_bottle(rng, bx, by)]


def scenario_drop(seed: int = 0, fps: int = FPS, duration_s: float = 9.0) -> Iterator[Frame]:
    """A person walks past carrying a bottle, sets it down and walks away."""
    rng = np.random.default_rng(seed)
    ground = None
    for i in range(int(duration_s * fps)):
        t = i / fps
        pcx, pcy = 60 + 200 * t, 300
        persons = [_person(rng, pcx, pcy)] if pcx < 640 + PERSON_W else []
        if t < 2.0:
            bx, by = pcx + 45, pcy - 20
        elif t < 2.3:  # bottle falls to the ground next to the feet
            if ground is None:
                ground = (pcx + 45, 425.0)
            frac = (t - 2.0) / 0.3
            bx, by = ground[0], (pcy - 20) + frac * (ground[1] - (pcy - 20))
        else:
            bx, by = ground
        yield t, persons, [_bottle(rng, bx, by)]


def scenario_bystander(seed: int = 0, fps: int = FPS, duration_s: float = 6.0) -> Iterator[Frame]:
    """Pre-existing litter on the ground; a person merely walks past it.
    No event must be raised."""
    rng = np.random.default_rng(seed)
    for i in range(int(duration_s * fps)):
        t = i / fps
        pcx = 60 + 130 * t
        persons = [_person(rng, pcx, 300)] if pcx < 640 + PERSON_W else []
        yield t, persons, [_bottle(rng, 300, 430)]


def run_scenario(frames, analysis_cfg=None) -> List[LitterEvent]:
    analyzer = LitterAnalyzer(analysis_cfg)
    events: List[LitterEvent] = []
    for t, persons, litters in frames:
        events += analyzer.step(t, persons, litters, FRAME_SIZE).events
    return events
