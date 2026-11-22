"""
Suspicious Activity Object Detection Module

Handles two critical CCTV surveillance scenarios:
1. Unattended / Abandoned Objects: Stationary luggage (backpack, handbag, suitcase)
   left behind without an owner in proximity for > T seconds.
2. Dangerous / Prohibited Objects: Instant detection of weapons or prohibited items
   (e.g., knife, scissors, baseball bat) and associating them with nearby tracked persons.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.config import SuspiciousObjectConfig
from src.tracking.tracker import TrackedPerson


@dataclass
class SuspiciousObjectEvent:
    track_id: int
    object_type: str
    timestamp: float
    confidence: float
    position: Tuple[float, float]
    event_type: str = "unattended_object"   # "unattended_object" | "prohibited_object"
    duration_seconds: Optional[float] = None
    associated_person_id: Optional[int] = None

    def to_dict(self):
        d = {
            "event_type": self.event_type,
            "track_id": self.track_id,
            "object_type": self.object_type,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "position": list(self.position),
        }
        if self.duration_seconds is not None:
            d["duration_seconds"] = round(self.duration_seconds, 1)
        if self.associated_person_id is not None:
            d["associated_person_id"] = self.associated_person_id
        return d


@dataclass
class _UnattendedState:
    first_unattended_time: Optional[float] = None
    last_seen_time: Optional[float] = None
    stationary_position: Optional[Tuple[float, float]] = None
    confirmed: bool = False


@dataclass
class _DangerousState:
    consecutive_frames: int = 0
    confirmed: bool = False
    last_seen_time: float = 0.0


class SuspiciousObjectDetector:
    """
    Stateful detector monitoring both unattended luggage and dangerous/prohibited items.
    """

    def __init__(self, cfg: SuspiciousObjectConfig):
        self.cfg = cfg
        self._unattended_states: Dict[int, _UnattendedState] = {}
        self._dangerous_states: Dict[int, _DangerousState] = {}

    def update(
        self,
        objects: List[TrackedPerson],
        people: List[TrackedPerson],
        timestamp: float,
    ) -> List[SuspiciousObjectEvent]:
        events: List[SuspiciousObjectEvent] = []
        if not self.cfg.enabled:
            return events

        # -------------------------------------------------------------
        # Part 1: Dangerous / Prohibited Object Detection
        # -------------------------------------------------------------
        if getattr(self.cfg, "dangerous_enabled", True):
            dangerous_classes = set(getattr(self.cfg, "dangerous_classes", ["knife", "scissors", "baseball bat"]))
            confirm_needed = getattr(self.cfg, "dangerous_confirm_frames", 2)

            active_dangerous_ids = set()

            for obj in objects:
                if obj.class_name in dangerous_classes:
                    active_dangerous_ids.add(obj.track_id)
                    dstate = self._dangerous_states.setdefault(obj.track_id, _DangerousState())
                    dstate.consecutive_frames += 1
                    dstate.last_seen_time = timestamp

                    if dstate.consecutive_frames >= confirm_needed and not dstate.confirmed:
                        dstate.confirmed = True

                        # Find closest person within association proximity
                        assoc_person_id = None
                        min_dist = 120.0  # px proximity
                        for p in people:
                            if p.class_name == "person":
                                dist = (
                                    (obj.center[0] - p.center[0]) ** 2 +
                                    (obj.center[1] - p.center[1]) ** 2
                                ) ** 0.5
                                if dist < min_dist:
                                    min_dist = dist
                                    assoc_person_id = p.track_id

                        events.append(SuspiciousObjectEvent(
                            track_id=obj.track_id,
                            object_type=obj.class_name,
                            timestamp=timestamp,
                            confidence=obj.confidence,
                            position=obj.bottom_center,
                            event_type="prohibited_object",
                            associated_person_id=assoc_person_id,
                        ))

            # Prune inactive dangerous states
            for tid in list(self._dangerous_states.keys()):
                if tid not in active_dangerous_ids or (timestamp - self._dangerous_states[tid].last_seen_time) > 3.0:
                    del self._dangerous_states[tid]

        # -------------------------------------------------------------
        # Part 2: Unattended / Abandoned Object Detection
        # -------------------------------------------------------------
        if getattr(self.cfg, "unattended_enabled", True):
            unattended_classes = set(getattr(self.cfg, "unattended_classes", getattr(self.cfg, "classes_to_watch", ["backpack", "handbag", "suitcase"])))
            watched_items = [obj for obj in objects if obj.class_name in unattended_classes]
            active_item_ids = {item.track_id for item in watched_items}

            # Clear state for lost items
            for tid in list(self._unattended_states.keys()):
                if tid not in active_item_ids:
                    del self._unattended_states[tid]

            for item in watched_items:
                item_pos = item.bottom_center
                state = self._unattended_states.setdefault(item.track_id, _UnattendedState())
                state.last_seen_time = timestamp

                # Check if any person is within owner_proximity_px
                has_nearby_person = False
                for person in people:
                    if person.class_name != "person":
                        continue
                    p_pos = person.bottom_center
                    dist = ((item_pos[0] - p_pos[0]) ** 2 + (item_pos[1] - p_pos[1]) ** 2) ** 0.5
                    if dist <= self.cfg.owner_proximity_px:
                        has_nearby_person = True
                        break

                if has_nearby_person:
                    # Owner is present, reset unattended timer
                    state.first_unattended_time = None
                    state.stationary_position = item_pos
                else:
                    # Item is unattended
                    if state.first_unattended_time is None:
                        state.first_unattended_time = timestamp
                        state.stationary_position = item_pos

                    unattended_duration = timestamp - state.first_unattended_time

                    if (unattended_duration >= self.cfg.time_threshold_seconds and not state.confirmed):
                        state.confirmed = True
                        events.append(SuspiciousObjectEvent(
                            track_id=item.track_id,
                            object_type=item.class_name,
                            timestamp=timestamp,
                            confidence=item.confidence,
                            position=item_pos,
                            event_type="unattended_object",
                            duration_seconds=unattended_duration,
                        ))

        return events

    def is_unattended(self, track_id: int) -> bool:
        state = self._unattended_states.get(track_id)
        return state.confirmed if state else False

    def is_dangerous(self, track_id: int) -> bool:
        state = self._dangerous_states.get(track_id)
        return state.confirmed if state else False
