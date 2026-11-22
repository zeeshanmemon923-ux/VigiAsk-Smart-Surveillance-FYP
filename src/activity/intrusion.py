"""
Restricted-Area Intrusion Detection (Proposal Section 4.4)

Operator/administrator defines restricted zones as polygons (via
tests/zone_editor.py -> data/zones.json). Every frame, each tracked
person's foot-point is checked against every zone. A person must be
inside a zone for `confirm_frames` consecutive frames before an
intrusion event fires — this absorbs a single jittery/wrong detection
instead of alarming on it.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.config import IntrusionConfig
from src.tracking.tracker import TrackedPerson


@dataclass
class Zone:
    name: str
    polygon: List[Tuple[float, float]]  # ordered points, in frame coords

    def contains(self, point: Tuple[float, float]) -> bool:
        pts = np.array(self.polygon, dtype=np.int32)
        result = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False)
        return result >= 0


@dataclass
class IntrusionEvent:
    track_id: int
    zone_name: str
    timestamp: float
    confidence: float
    position: Tuple[float, float]

    def to_dict(self):
        return {
            "event_type": "restricted_area_intrusion",
            "track_id": self.track_id,
            "zone": self.zone_name,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "position": list(self.position),
        }


def load_zones(zones_file: str) -> List[Zone]:
    path = Path(zones_file)
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return [Zone(name=z["name"], polygon=[tuple(p) for p in z["polygon"]])
            for z in data.get("zones", [])]


@dataclass
class _TrackZoneState:
    """Tracks how many consecutive frames a given track has been inside a given zone."""
    consecutive_frames: int = 0
    confirmed: bool = False


class IntrusionDetector:
    """
    Stateful detector: call `update()` every frame with the current
    tracked people; get back newly confirmed IntrusionEvents (fires
    once per track/zone entry, not every frame the person stays inside).
    """

    def __init__(self, cfg: IntrusionConfig):
        self.cfg = cfg
        self.zones: List[Zone] = load_zones(cfg.zones_file)
        # state[(track_id, zone_name)] = _TrackZoneState
        self._state: Dict[Tuple[int, str], _TrackZoneState] = {}

    def update(self, people: List[TrackedPerson], timestamp: float) -> List[IntrusionEvent]:
        events: List[IntrusionEvent] = []
        if not self.zones:
            return events

        active_keys = set()

        for person in people:
            point = person.bottom_center
            for zone in self.zones:
                key = (person.track_id, zone.name)
                inside = zone.contains(point)

                if inside:
                    active_keys.add(key)
                    state = self._state.setdefault(key, _TrackZoneState())
                    state.consecutive_frames += 1

                    if (state.consecutive_frames >= self.cfg.confirm_frames
                            and not state.confirmed):
                        state.confirmed = True
                        events.append(IntrusionEvent(
                            track_id=person.track_id,
                            zone_name=zone.name,
                            timestamp=timestamp,
                            confidence=person.confidence,
                            position=point,
                        ))

        # reset state for any (track, zone) pair that's no longer active this frame
        for key in list(self._state.keys()):
            if key not in active_keys:
                del self._state[key]

        return events

    def currently_inside(self) -> List[Tuple[int, str]]:
        """(track_id, zone_name) pairs currently confirmed as inside a zone."""
        return [k for k, s in self._state.items() if s.confirmed]
