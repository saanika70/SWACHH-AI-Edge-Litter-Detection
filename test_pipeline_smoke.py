"""End-to-end smoke test: fake camera + fake YOLO replaying a synthetic throw."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from swachh_ai import pipeline as pl  # noqa: E402
from swachh_ai.camera import Frame  # noqa: E402
from swachh_ai.simulation import scenario_throw  # noqa: E402
from swachh_ai.store import EventStore  # noqa: E402


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    frames = list(scenario_throw())
    state = {"i": 0}

    class FakeCamera:
        is_file = True

        def __init__(self, cfg):
            pass

        def open(self):
            pass

        def close(self):
            pass

        def read(self, timeout=1.0):
            i = state["i"]
            if i >= len(frames):
                return None
            state["i"] += 1
            return Frame(np.zeros((480, 640, 3), np.uint8), frames[i][0], i)

    class FakeDetector:
        def __init__(self, classes=None, **kw):
            self.is_person = classes == [0]

        def __call__(self, img):
            _, persons, litters = frames[state["i"] - 1]
            return persons if self.is_person else litters

    monkeypatch.setattr(pl, "Camera", FakeCamera)
    monkeypatch.setattr(pl, "YoloDetector", FakeDetector)

    cfg = {
        "device": {"id": "test-dev"},
        "models": {"litter": {"weights": "x"}, "person": {"weights": "y"}},
        "inference": {"active_fps": 100},
        "hardware": {"backend": "mock"},
        "storage": {"db_path": str(tmp_path / "e.db"), "snapshot_dir": str(tmp_path / "snaps")},
        "iot": {},
    }
    pl.SwachhPipeline(cfg, force_mock_hw=True, disable_notify=True).run()

    rows = EventStore(str(tmp_path / "e.db")).recent()
    assert [r["kind"] for r in rows] == ["throw"]
    assert rows[0]["device_id"] == "test-dev"
    assert Path(rows[0]["snapshot"]).is_file()
