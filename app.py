"""Tiny Flask dashboard for the waste-monitoring side of the project.

    SWACHH_DB=data/events.db SWACHH_SNAPSHOTS=data/snapshots python dashboard/app.py

Shows recent littering events with (head-blurred) evidence snapshots and
per-day counts.  Bind to localhost / put behind a reverse proxy with auth
before exposing it to the internet.
"""
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swachh_ai.store import EventStore  # noqa: E402

DB = os.environ.get("SWACHH_DB", "data/events.db")
SNAPS = Path(os.environ.get("SWACHH_SNAPSHOTS", "data/snapshots")).resolve()

app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent / "templates"))
store = EventStore(DB)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/events")
def events():
    rows = store.recent(50)
    for r in rows:
        r["snapshot_url"] = f"/snapshots/{Path(r['snapshot']).name}" if r.get("snapshot") else None
    return jsonify(rows)


@app.get("/api/stats")
def stats():
    return jsonify(store.daily_counts(14))


@app.get("/snapshots/<path:name>")
def snapshot(name):
    return send_from_directory(SNAPS, name)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)))
