from swachh_ai.accumulation import AccumulationMonitor
from swachh_ai.types import Detection

SIZE = (640, 480)


def litter(n, x0=50):
    return [Detection("plastic_bag", 0.8, (x0 + 60 * i, 400, x0 + 60 * i + 30, 430)) for i in range(n)]


def test_needs_consecutive_scans_and_respects_cooldown():
    mon = AccumulationMonitor(threshold=5, consecutive_scans=2, cooldown_s=100)
    assert mon.update(0, litter(6), [], SIZE) is None  # streak 1
    ev = mon.update(60, litter(6), [], SIZE)
    assert ev is not None and ev.kind == "accumulation" and ev.extra["count"] == 6
    assert mon.update(120, litter(6), [], SIZE) is None  # cooldown
    assert mon.update(200, litter(6), [], SIZE) is not None


def test_below_threshold_resets_streak():
    mon = AccumulationMonitor(threshold=5, consecutive_scans=2)
    mon.update(0, litter(6), [], SIZE)
    mon.update(60, litter(2), [], SIZE)
    assert mon.update(120, litter(6), [], SIZE) is None


def test_ignores_zones_and_litter_near_people():
    mon = AccumulationMonitor(threshold=1, exclusion_zones=[[0.0, 0.8, 1.0, 1.0]])
    assert mon.count_ground_litter(litter(4), [], SIZE) == 0
    mon2 = AccumulationMonitor(threshold=1)
    person = Detection("person", 0.9, (30, 100, 140, 420))
    assert mon2.count_ground_litter(litter(1), [person], SIZE) == 0
