from swachh_ai.tracker import Tracker
from swachh_ai.types import Detection


def det(cx, cy, label="bottle"):
    return Detection(label, 0.9, (cx - 10, cy - 10, cx + 10, cy + 10))


def test_keeps_identity_for_moving_object():
    tr = Tracker(max_dist_px=100)
    ids = set()
    for i in range(10):
        tracks = tr.update([det(100 + 30 * i, 200)], i * 0.1)
        ids |= {t.id for t in tracks}
    assert ids == {1}


def test_label_mismatch_creates_new_track():
    tr = Tracker()
    tr.update([det(100, 100, "bottle")], 0.0)
    tracks = tr.update([det(102, 100, "can")], 0.1)
    assert {t.label for t in tracks} == {"bottle", "can"}


def test_track_expires():
    tr = Tracker(max_age_s=1.0)
    tr.update([det(100, 100)], 0.0)
    assert tr.update([], 2.0) == []


def test_velocity_estimate():
    tr = Tracker()
    for i in range(6):
        tracks = tr.update([det(100 + 50 * i, 100)], i * 0.1)
    vx, vy = tracks[0].velocity()
    assert abs(vx - 500) < 1e-6 and abs(vy) < 1e-6
