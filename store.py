"""SQLite event log used by the edge device and the dashboard."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import LitterEvent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    device_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    person_id INTEGER,
    litter_id INTEGER,
    label TEXT,
    confidence REAL,
    snapshot TEXT,
    extra TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
"""


class EventStore:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)

    def add_event(self, device_id: str, ev: LitterEvent, snapshot: Optional[str]) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (ts, device_id, kind, person_id, litter_id, label, confidence, snapshot, extra)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    ev.wall_time,
                    device_id,
                    ev.kind,
                    ev.person_id,
                    ev.litter_id,
                    ev.label,
                    ev.confidence,
                    snapshot,
                    json.dumps(ev.extra),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def daily_counts(self, days: int = 14) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT date(ts,'unixepoch','localtime') AS day, kind, COUNT(*) AS n FROM events "
                "WHERE ts >= strftime('%s','now', ?) GROUP BY day, kind ORDER BY day",
                (f"-{int(days)} days",),
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
