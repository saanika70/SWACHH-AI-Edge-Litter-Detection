"""The edge pipeline: sensors -> camera -> YOLO -> analysis -> actuators -> IoT.

Power-aware state machine
-------------------------
IDLE    no presence: no inference (only a slow "patrol" scan for litter
        accumulation), dim/off LED.
ACTIVE  PIR / ultrasonic / person detected: inference at ``active_fps``,
        LED brightens.
WATCH   a person is carrying litter: LED goes to full brightness (deterrent).
ALERT   throw / drop detected: LED strobe + buzzer + snapshot + IoT alert.
"""
from __future__ import annotations

import logging
import signal
import time
from typing import Any, Dict, List, Optional

from .accumulation import AccumulationMonitor
from .analyzer import LitterAnalyzer
from .camera import Camera, Frame
from .detector import YoloDetector
from .hardware import LightingController, Mode, create_hardware
from .iot import Notifier
from .store import EventStore
from .types import Detection, LitterEvent
from .vis import SnapshotWriter, draw_overlay

log = logging.getLogger("swachh.pipeline")


def _cpu_temp_c() -> Optional[float]:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as fh:
            return round(int(fh.read().strip()) / 1000.0, 1)
    except Exception:
        return None


class SwachhPipeline:
    def __init__(self, cfg: Dict[str, Any], show: bool = False, force_mock_hw: bool = False,
                 disable_notify: bool = False):
        self.cfg = cfg
        self.show = show
        dev = cfg.get("device", {})
        self.device_id = dev.get("id", "swachh-001")
        self.location = dev.get("location", "")

        self.camera = Camera(cfg.get("camera", {}))

        models = cfg["models"]
        self.litter_det = YoloDetector(**models["litter"])
        pcfg = dict(models.get("person") or {})
        self.person_every = max(int(pcfg.pop("every_n_frames", 1)), 1)
        # person weights == null -> use the "person" class of the litter model
        self.person_det = YoloDetector(classes=[0], **pcfg) if pcfg.get("weights") else None

        analysis = cfg.get("analysis", {})
        self.analyzer = LitterAnalyzer(analysis)
        acc = cfg.get("accumulation", {})
        self.accumulation = AccumulationMonitor(
            threshold=acc.get("threshold", 5),
            consecutive_scans=acc.get("consecutive_scans", 2),
            cooldown_s=acc.get("cooldown_s", 1800),
            near_ratio=analysis.get("near_ratio", 0.6),
            exclusion_zones=analysis.get("exclusion_zones", []),
        )

        self.hw = create_hardware(cfg.get("hardware", {}), force_mock=force_mock_hw)
        self.lighting = LightingController(self.hw, cfg.get("lighting", {}))
        self.buzzer_cfg = cfg.get("buzzer", {})

        st = cfg.get("storage", {})
        self.store = EventStore(st.get("db_path", "data/events.db"))
        self.snapshots = SnapshotWriter(
            st.get("snapshot_dir", "data/snapshots"),
            st.get("max_snapshots", 500),
            st.get("blur_people", True),
        )

        iot_cfg = cfg.get("iot", {})
        if disable_notify:
            iot_cfg = {}
        self.notifier = Notifier(iot_cfg, self.device_id, self.location)

        inf = cfg.get("inference", {})
        self.active_interval = 1.0 / float(inf.get("active_fps", 6))
        self.active_hold_s = float(inf.get("active_hold_s", 20))
        self.idle_poll_s = float(inf.get("idle_poll_s", 0.1))
        self.patrol_interval_s = float(inf.get("patrol_interval_s", 60))
        self.telemetry_interval_s = float(inf.get("telemetry_interval_s", 60))

        self._stop = False
        self._last_presence = -1e9
        self._next_patrol = 0.0
        self._next_telemetry = time.monotonic() + self.telemetry_interval_s
        self._last_proc_ts = -1e9
        self._frame_idx = 0
        self._fps = 0.0
        self._last_persons: List[Detection] = []
        self._mode = Mode.IDLE

    # ------------------------------------------------------------------ run
    def stop(self, *_: Any) -> None:
        self._stop = True

    def run(self) -> None:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        self.camera.open()
        log.info("SWACHH-AI running (device=%s)", self.device_id)
        try:
            while not self._stop:
                now = time.monotonic()
                presence = self.hw.read_presence()
                dark = self.hw.is_dark()
                if presence.present:
                    self._last_presence = now
                active = (now - self._last_presence) < self.active_hold_s

                if not active:
                    self._idle_step(now, dark)
                    continue

                frame = self.camera.read(timeout=1.0)
                if frame is None:
                    if self.camera.is_file:
                        log.info("End of video")
                        break
                    continue
                if frame.ts - self._last_proc_ts < self.active_interval:
                    continue
                self._last_proc_ts = frame.ts
                self._process(frame, dark, now)
                if self.show and self._window_closed():
                    break
        finally:
            self.close()

    def close(self) -> None:
        log.info("Shutting down")
        for fn in (self.camera.close, self.notifier.close, self.hw.close, self.store.close):
            try:
                fn()
            except Exception:
                pass
        if self.show:
            import cv2

            cv2.destroyAllWindows()

    # ---------------------------------------------------------------- steps
    def _idle_step(self, now: float, dark: bool) -> None:
        self._mode = Mode.IDLE
        self.lighting.update(Mode.IDLE, dark)
        if now >= self._next_patrol:
            self._next_patrol = now + self.patrol_interval_s
            frame = self.camera.read(timeout=2.0)
            if frame is not None:
                litter, persons = self._detect(frame.image, run_person=True)
                self._scan_accumulation(frame, litter, persons or [])
        self._maybe_telemetry(now)
        time.sleep(self.idle_poll_s)

    def _detect(self, img, run_person: bool):
        litter = self.litter_det(img)
        if self.person_det is None:
            persons = [d for d in litter if d.label == "person"]
            litter = [d for d in litter if d.label != "person"]
        elif run_person:
            persons = self.person_det(img)
        else:
            persons = None
        return litter, persons

    def _process(self, frame: Frame, dark: bool, now: float) -> None:
        t0 = time.perf_counter()
        img = frame.image
        h, w = img.shape[:2]
        self._frame_idx += 1
        run_person = (self._frame_idx % self.person_every) == 0 or self.person_every == 1

        litter, persons = self._detect(img, run_person)
        if persons is not None:
            self._last_persons = persons
            if persons:
                self._last_presence = now  # keep the system awake while someone is in view

        result = self.analyzer.step(frame.ts, persons, litter, (w, h))

        self._mode = Mode.WATCH if result.watch_person_ids else Mode.ACTIVE
        self.lighting.update(self._mode, dark)

        p_tracks = list(self.analyzer.person_tracker.tracks.values())
        l_tracks = [t for t in self.analyzer.litter_tracker.tracks.values() if t.updated]

        for ev in result.events:
            self._handle_event(ev, img, p_tracks, l_tracks)

        if now >= self._next_patrol:
            self._next_patrol = now + self.patrol_interval_s
            self._scan_accumulation(frame, litter, self._last_persons)

        dt = time.perf_counter() - t0
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt
        self._maybe_telemetry(now)

        if self.show:
            import cv2

            vis = draw_overlay(
                img,
                [(t.id, t.box) for t in p_tracks if t.updated],
                [(t.id, t.label, t.box) for t in l_tracks],
                f"{self._mode.value.upper()}  {self._fps:.1f} FPS  ground litter: {self.accumulation.last_count}",
                result.watch_person_ids,
            )
            cv2.imshow("SWACHH-AI", vis)

    def _scan_accumulation(self, frame: Frame, litter: List[Detection], persons: List[Detection]) -> None:
        h, w = frame.image.shape[:2]
        ev = self.accumulation.update(frame.ts, litter, persons, (w, h))
        if ev is not None:
            boxes = [(d.label, d.box) for d in litter]
            self._handle_event(ev, frame.image, [], [], extra_boxes=boxes)

    def _handle_event(self, ev: LitterEvent, img, p_tracks, l_tracks, extra_boxes=None) -> None:
        log.warning("EVENT %s label=%s person=%s extra=%s", ev.kind, ev.label, ev.person_id, ev.extra)
        if ev.kind in ("throw", "drop"):
            self._mode = Mode.ALERT
            self.lighting.alert()
            b = self.buzzer_cfg
            self.hw.beep(int(b.get("alert_beeps", 3)), float(b.get("beep_on_s", 0.2)), float(b.get("beep_off_s", 0.15)))
        person_boxes = [t.box for t in p_tracks if t.updated]
        litter_boxes = extra_boxes if extra_boxes is not None else [(t.label, t.box) for t in l_tracks]
        path = None
        try:
            path = self.snapshots.save(img, ev, person_boxes, litter_boxes, self.device_id)
            ev.extra["snapshot"] = str(path)
        except Exception as exc:
            log.warning("Snapshot failed: %s", exc)
        self.store.add_event(self.device_id, ev, str(path) if path else None)
        self.notifier.send_event(ev, path)

    def _maybe_telemetry(self, now: float) -> None:
        if now < self._next_telemetry:
            return
        self._next_telemetry = now + self.telemetry_interval_s
        self.notifier.send_telemetry(
            {
                "mode": self._mode.value,
                "fps": round(self._fps, 2),
                "ground_litter": self.accumulation.last_count,
                "cpu_temp_c": _cpu_temp_c(),
                "battery": self.hw.battery(),
            }
        )

    @staticmethod
    def _window_closed() -> bool:
        import cv2

        return (cv2.waitKey(1) & 0xFF) in (ord("q"), 27)
