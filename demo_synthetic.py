"""Run the analysis engine on synthetic scenarios - no camera, model or GPIO needed.

    python scripts/demo_synthetic.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swachh_ai.simulation import run_scenario, scenario_bystander, scenario_drop, scenario_throw  # noqa: E402

for name, fn, expect in [
    ("person throws a bottle", scenario_throw, "throw"),
    ("person puts a bottle down and walks away", scenario_drop, "drop"),
    ("person walks past existing litter", scenario_bystander, None),
]:
    events = run_scenario(fn())
    kinds = [e.kind for e in events]
    status = "OK " if (kinds == [expect] if expect else not kinds) else "FAIL"
    detail = ", ".join(f"{e.kind}@{e.timestamp:.1f}s" for e in events) or "no event"
    print(f"[{status}] {name:<45} -> {detail}")
