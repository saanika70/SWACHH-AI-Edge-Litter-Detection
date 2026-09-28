"""Non-blocking IoT notifications: MQTT, Telegram (with snapshot), webhook.

Every channel is optional and failure-tolerant; sending happens on a worker
thread so network problems can never stall the vision loop.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..types import LitterEvent

log = logging.getLogger("swachh.iot")

_TITLES = {
    "throw": "Littering detected (item thrown)",
    "drop": "Littering detected (item dropped)",
    "accumulation": "Litter accumulation - cleaning needed",
}


class Notifier:
    def __init__(self, cfg: Dict[str, Any], device_id: str, location: str = ""):
        self.cfg = cfg or {}
        self.device_id = device_id
        self.location = location
        self.min_interval = float(self.cfg.get("min_interval_s", 5))
        self._last_sent: Dict[str, float] = {}
        self._q: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=200)
        self._mqtt = None
        self._prefix = ""
        self._init_mqtt()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # -- setup ------------------------------------------------------------
    def _init_mqtt(self) -> None:
        m = self.cfg.get("mqtt", {}) or {}
        if not m.get("enabled"):
            return
        try:
            import paho.mqtt.client as mqtt

            self._prefix = f"{m.get('topic_prefix', 'swachh')}/{self.device_id}"
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"{self.device_id}")
            if m.get("username"):
                client.username_pw_set(m["username"], m.get("password") or None)
            if m.get("tls"):
                client.tls_set()
            client.will_set(f"{self._prefix}/status", "offline", qos=1, retain=True)

            def on_connect(c, userdata, flags, reason_code, properties=None):
                log.info("MQTT connected (%s)", reason_code)
                c.publish(f"{self._prefix}/status", "online", qos=1, retain=True)

            client.on_connect = on_connect
            client.connect_async(m.get("host", "localhost"), int(m.get("port", 1883)), keepalive=60)
            client.loop_start()
            self._mqtt = client
        except Exception as exc:
            log.warning("MQTT disabled: %s", exc)

    # -- public API -------------------------------------------------------
    def send_event(self, ev: LitterEvent, snapshot: Optional[Path] = None) -> None:
        now = time.monotonic()
        if now - self._last_sent.get(ev.kind, -1e9) < self.min_interval:
            return
        self._last_sent[ev.kind] = now
        payload = {
            "device_id": self.device_id,
            "location": self.location,
            "type": ev.kind,
            "label": ev.label,
            "confidence": round(ev.confidence, 3),
            "timestamp": ev.wall_time,
            "extra": ev.extra,
            "snapshot": str(snapshot) if snapshot else None,
        }
        self._enqueue(("event", payload, snapshot))

    def send_telemetry(self, data: Dict[str, Any]) -> None:
        self._enqueue(("telemetry", {"device_id": self.device_id, "timestamp": time.time(), **data}, None))

    def close(self) -> None:
        self._enqueue(None)
        self._thread.join(timeout=3.0)
        if self._mqtt is not None:
            try:
                self._mqtt.publish(f"{self._prefix}/status", "offline", qos=1, retain=True).wait_for_publish(1.0)
            except Exception:
                pass
            self._mqtt.loop_stop()
            self._mqtt.disconnect()

    # -- internals --------------------------------------------------------
    def _enqueue(self, item) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            log.warning("Notification queue full - dropping message")

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            kind, payload, snapshot = item
            try:
                if self._mqtt is not None:
                    self._mqtt.publish(f"{self._prefix}/{kind}", json.dumps(payload), qos=1)
                if kind == "event":
                    self._telegram(payload, snapshot)
                    self._webhook(payload)
            except Exception as exc:
                log.warning("Notification failed: %s", exc)

    def _telegram(self, payload: Dict[str, Any], snapshot: Optional[Path]) -> None:
        t = self.cfg.get("telegram", {}) or {}
        if not (t.get("enabled") and t.get("token") and t.get("chat_id")):
            return
        import requests

        when = time.strftime("%d %b %Y %H:%M:%S", time.localtime(payload["timestamp"]))
        caption = f"{_TITLES.get(payload['type'], payload['type'])}\n{payload['location']} ({payload['device_id']})\n{when}"
        base = f"https://api.telegram.org/bot{t['token']}"
        if snapshot and Path(snapshot).is_file():
            with open(snapshot, "rb") as fh:
                requests.post(f"{base}/sendPhoto", data={"chat_id": t["chat_id"], "caption": caption},
                              files={"photo": fh}, timeout=15)
        else:
            requests.post(f"{base}/sendMessage", data={"chat_id": t["chat_id"], "text": caption}, timeout=10)

    def _webhook(self, payload: Dict[str, Any]) -> None:
        w = self.cfg.get("webhook", {}) or {}
        if not (w.get("enabled") and w.get("url")):
            return
        import requests

        requests.post(w["url"], json=payload, timeout=10)
