"""
Weapon event logic.

A single-frame weapon detection is unreliable (phones look like guns, umbrellas
like rifles), so an event only fires when the same weapon class is seen in at
least `confirm_hits` of the last `confirm_window` inference runs. After firing,
it re-arms once the class has been absent for `rearm_runs` runs. The
AlertManager cooldown still applies on top of this.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from src.config import WeaponConfig
from src.detection.weapon_detector import WeaponDetection


@dataclass
class WeaponEvent:
    weapon_class: str
    confidence: float
    timestamp: float
    position: Tuple[float, float]
    holder_track_id: Optional[int] = None

    def to_dict(self):
        return {
            "event_type": "weapon_detected",
            "object_type": self.weapon_class,
            "track_id": self.holder_track_id,   # person holding / nearest to it
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "position": list(self.position),
        }


def _holder(det: WeaponDetection, people) -> Optional[int]:
    """Person whose box contains the weapon centre, else None."""
    cx, cy = det.center
    best, best_area = None, None
    for p in people:
        x1, y1, x2, y2 = p.bbox_xyxy
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            area = (x2 - x1) * (y2 - y1)
            if best_area is None or area < best_area:   # smallest box = most specific
                best, best_area = p.track_id, area
    return best


class WeaponEventDetector:
    def __init__(self, cfg: WeaponConfig):
        self.cfg = cfg
        self._hits: Dict[str, Deque[bool]] = {}
        self._active: Dict[str, bool] = {}
        self._absent: Dict[str, int] = {}

    def update(self, detections: List[WeaponDetection], people, timestamp: float) -> List[WeaponEvent]:
        """Call ONLY on frames where WeaponDetector.ran_this_frame is True."""
        events: List[WeaponEvent] = []
        by_class: Dict[str, WeaponDetection] = {}
        for d in detections:
            if d.class_name not in by_class or d.confidence > by_class[d.class_name].confidence:
                by_class[d.class_name] = d

        for cname in set(by_class) | set(self._hits):
            hist = self._hits.setdefault(cname, deque(maxlen=self.cfg.confirm_window))
            seen = cname in by_class
            hist.append(seen)

            if seen:
                self._absent[cname] = 0
            else:
                self._absent[cname] = self._absent.get(cname, 0) + 1
                if self._absent[cname] >= self.cfg.rearm_runs:
                    self._active[cname] = False

            if seen and not self._active.get(cname, False) and sum(hist) >= self.cfg.confirm_hits:
                d = by_class[cname]
                self._active[cname] = True
                events.append(WeaponEvent(cname, d.confidence, timestamp, d.center, _holder(d, people)))
        return events

    def is_active(self, weapon_class: str) -> bool:
        return self._active.get(weapon_class, False)