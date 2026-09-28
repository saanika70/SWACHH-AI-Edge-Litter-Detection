"""Person-litter proximity analysis and throw / drop detection.

The engine is fed tracked persons and tracked litter items every analysed frame
and emits :class:`LitterEvent` objects.  All distances are normalised by the
person's bounding-box height and all speeds are expressed in *person heights
per second*, so the thresholds are camera-distance independent.

Event logic
-----------
* **held**    litter that spent >= ``hold_min_s`` within ``near_ratio`` of a
              person (or first appeared next to one).
* **carried** litter that was held *and* moved together with the person.
* **THROW**   a held item leaves the person's neighbourhood
              (>= ``release_ratio``) faster than ``throw_speed`` and moves
              away from the person.
* **DROP**    a carried item becomes stationary for ``drop_stationary_s`` and
              its carrier has left (>= ``leave_ratio`` away or no longer seen).
* **WATCH**   a person is currently carrying litter (used to brighten the
              light as a deterrent - no alarm).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .geometry import box_center, in_zones, proximity_ratio
from .tracker import Track
from .types import LitterEvent

VEL_WINDOW_S = 0.4


@dataclass
class AnalysisConfig:
    near_ratio: float = 0.6
    release_ratio: float = 0.9
    leave_ratio: float = 1.0
    hold_min_s: float = 0.4
    lookback_s: float = 1.5
    carry_speed: float = 0.25
    throw_speed: float = 1.5
    stationary_speed: float = 0.15
    drop_stationary_s: float = 2.0
    min_track_hits: int = 3
    person_cooldown_s: float = 15.0
    exclusion_zones: list = field(default_factory=list)  # normalised [x1,y1,x2,y2]

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "AnalysisConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


@dataclass
class _State:
    near_person: Optional[int] = None
    near_since: Optional[float] = None
    last_near_t: float = -1e9
    held_by: Optional[int] = None
    carried_by: Optional[int] = None
    ref_height: float = 0.0
    stationary_since: Optional[float] = None
    reported: bool = False


@dataclass
class EngineResult:
    events: List[LitterEvent] = field(default_factory=list)
    watch_person_ids: Set[int] = field(default_factory=set)
    near_pairs: List[Tuple[int, int, float]] = field(default_factory=list)  # (person, litter, ratio)


class LitterEventEngine:
    def __init__(self, cfg: Optional[AnalysisConfig] = None):
        self.cfg = cfg or AnalysisConfig()
        self._states: Dict[int, _State] = {}
        self._person_last_event: Dict[Optional[int], float] = {}

    def process(
        self,
        t: float,
        persons: Sequence[Track],
        litters: Sequence[Track],
        frame_size: Tuple[int, int],
    ) -> EngineResult:
        cfg = self.cfg
        fw, fh = frame_size
        result = EngineResult()
        person_by_id = {p.id: p for p in persons}
        person_boxes = {p.id: p.box_at(t) for p in persons}

        # forget litter tracks that no longer exist
        alive = {lt.id for lt in litters}
        for lid in [k for k in self._states if k not in alive]:
            del self._states[lid]

        for lt in litters:
            if not lt.updated:
                continue
            if in_zones(lt.center, cfg.exclusion_zones, fw, fh):  # e.g. a dustbin
                self._states.pop(lt.id, None)
                continue

            st = self._states.setdefault(lt.id, _State())

            # nearest person (in person-heights)
            best: Optional[Tuple[int, float]] = None
            for pid, pb in person_boxes.items():
                r = proximity_ratio(lt.box, pb)
                if best is None or r < best[1]:
                    best = (pid, r)
            near = best is not None and best[1] <= cfg.near_ratio

            ref = person_by_id[best[0]].height if near else (st.ref_height or fh * 0.5)
            vx, vy = lt.velocity(VEL_WINDOW_S)
            speed = math.hypot(vx, vy) / max(ref, 1.0)

            if near:
                pid, ratio = best  # type: ignore[misc]
                if st.near_person != pid:
                    st.near_person, st.near_since = pid, t
                st.last_near_t = t
                st.ref_height = ref
                near_for = t - (st.near_since if st.near_since is not None else t)
                if lt.hits == 1 or near_for >= cfg.hold_min_s:
                    st.held_by = pid
                if near_for >= cfg.hold_min_s and speed >= cfg.carry_speed:
                    st.carried_by = pid
                result.near_pairs.append((pid, lt.id, ratio))
                if st.carried_by == pid and not st.reported:
                    result.watch_person_ids.add(pid)
            else:
                st.near_person = st.near_since = None

            # ---- THROW --------------------------------------------------
            if (
                not st.reported
                and st.held_by is not None
                and lt.hits >= cfg.min_track_hits
                and (t - st.last_near_t) <= cfg.lookback_s
            ):
                holder = person_by_id.get(st.held_by)
                if holder is not None:
                    hb = person_boxes[holder.id]
                    hcx, hcy = box_center(hb)
                    away = vx * (lt.cx - hcx) + vy * (lt.cy - hcy) > 0
                    if (
                        speed >= cfg.throw_speed
                        and proximity_ratio(lt.box, hb) >= cfg.release_ratio
                        and away
                    ):
                        self._emit("throw", t, lt, holder.id, hb, st, result, speed)

            # ---- stationary bookkeeping ----------------------------------
            if speed < cfg.stationary_speed:
                if st.stationary_since is None:
                    st.stationary_since = t
            else:
                st.stationary_since = None

            # ---- DROP ----------------------------------------------------
            if (
                not st.reported
                and st.carried_by is not None
                and st.stationary_since is not None
                and (t - st.stationary_since) >= cfg.drop_stationary_s
                and lt.hits >= cfg.min_track_hits
                and not near
            ):
                carrier = person_by_id.get(st.carried_by)
                left = carrier is None or (
                    proximity_ratio(lt.box, person_boxes[carrier.id]) >= cfg.leave_ratio
                )
                if left:
                    pbox = person_boxes.get(st.carried_by)
                    self._emit("drop", t, lt, st.carried_by, pbox, st, result, speed)

        return result

    def _emit(self, kind, t, lt, pid, pbox, st, result, speed) -> None:
        st.reported = True
        last = self._person_last_event.get(pid)
        if pid is not None and last is not None and (t - last) < self.cfg.person_cooldown_s:
            return
        self._person_last_event[pid] = t
        result.events.append(
            LitterEvent(
                kind=kind,
                timestamp=t,
                person_id=pid,
                litter_id=lt.id,
                label=lt.label,
                confidence=float(lt.conf),
                litter_box=lt.box,
                person_box=pbox,
                extra={"speed_person_heights_per_s": round(float(speed), 2)},
            )
        )
