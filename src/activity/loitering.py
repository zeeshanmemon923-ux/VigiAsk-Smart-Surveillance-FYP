"""
Loitering Detection (Proposal Section 4.5)

Uses the same per-track position history the tracker already keeps
(TrackHistory from src/tracking/tracker.py). A track is "loitering"
once it has been present for longer than `time_threshold_seconds`
AND has not moved more than `movement_tolerance_px` from its recent
average position — i.e. present a long time but not just walking
through.

Fires once per track (won't re-alert every frame after confirmation)
until the track is lost and a new one starts.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.config import LoiteringConfig
from src.tracking.tracker import TrackedPerson, TrackHistory


@dataclass
class LoiteringEvent:
    track_id: int
    timestamp: float
    duration_seconds: float
    position: Tuple[float, float]

    def to_dict(self):
        return {
            "event_type": "loitering",
            "track_id": self.track_id,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "position": list(self.position),
        }


def _max_spread(positions: List[Tuple[float, float]]) -> float:
    """
    How far the track has wandered within its recent history window —
    max distance between any position and the window's average position.
    Cheap stand-in for 'has this person basically stayed in place'.
    """
    if len(positions) < 2:
        return 0.0
    avg_x = sum(p[0] for p in positions) / len(positions)
    avg_y = sum(p[1] for p in positions) / len(positions)
    return max(((p[0] - avg_x) ** 2 + (p[1] - avg_y) ** 2) ** 0.5 for p in positions)


class LoiteringDetector:
    """
    Stateful: call `update()` every frame with the tracker's current
    people + its histories dict. Returns newly confirmed LoiteringEvents.
    """

    def __init__(self, cfg: LoiteringConfig):
        self.cfg = cfg
        self._confirmed: Dict[int, bool] = {}

    def update(
        self,
        people: List[TrackedPerson],
        histories: Dict[int, TrackHistory],
        timestamp: float,
    ) -> List[LoiteringEvent]:
        events: List[LoiteringEvent] = []
        active_ids = {p.track_id for p in people}

        # clear confirmation state for tracks that disappeared, so if the
        # same physical person comes back later (new track_id) it can
        # re-trigger rather than being silently suppressed forever
        for tid in list(self._confirmed.keys()):
            if tid not in active_ids:
                del self._confirmed[tid]

        for person in people:
            hist = histories.get(person.track_id)
            if hist is None or hist.duration_seconds < self.cfg.time_threshold_seconds:
                continue
            if self._confirmed.get(person.track_id):
                continue  # already alerted for this track

            spread = _max_spread(hist.positions)
            if spread <= self.cfg.movement_tolerance_px:
                self._confirmed[person.track_id] = True
                events.append(LoiteringEvent(
                    track_id=person.track_id,
                    timestamp=timestamp,
                    duration_seconds=hist.duration_seconds,
                    position=person.bottom_center,
                ))

        return events

    def is_loitering(self, track_id: int) -> bool:
        return self._confirmed.get(track_id, False)
