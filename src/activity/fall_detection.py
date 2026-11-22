"""
Observable Fall Detection Module

Monitors posture collapse using bounding-box aspect ratio dynamics (Width / Height)
and vertical velocity trends across consecutive frames.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.config import FallDetectionConfig
from src.tracking.tracker import TrackedPerson, TrackHistory


@dataclass
class FallEvent:
    track_id: int
    timestamp: float
    aspect_ratio: float
    confidence: float
    position: Tuple[float, float]
    event_type: str = "possible_fall"

    def to_dict(self):
        return {
            "event_type": "possible_fall",
            "track_id": self.track_id,
            "timestamp": self.timestamp,
            "aspect_ratio": round(self.aspect_ratio, 2),
            "confidence": self.confidence,
            "position": list(self.position),
        }


@dataclass
class _FallTrackState:
    consecutive_low_profile: int = 0
    confirmed: bool = False
    triggered_time: float = 0.0


class FallDetector:
    """
    Evaluates tracked individuals for rapid downward motion followed by horizontal posture.
    Adheres strictly to objective observation without speculating on individual health.
    """

    def __init__(self, cfg: FallDetectionConfig):
        self.cfg = cfg
        self._states: Dict[int, _FallTrackState] = {}

    def update(
        self,
        people: List[TrackedPerson],
        histories: Dict[int, TrackHistory],
        timestamp: float,
    ) -> List[FallEvent]:
        events: List[FallEvent] = []
        if not self.cfg.enabled:
            return events

        active_ids = {p.track_id for p in people if p.class_name == "person"}

        # Clear tracks that are no longer active
        for tid in list(self._states.keys()):
            if tid not in active_ids:
                del self._states[tid]

        for person in people:
            if person.class_name != "person":
                continue

            hist = histories.get(person.track_id)
            if hist is None:
                continue

            state = self._states.setdefault(person.track_id, _FallTrackState())
            ar = person.aspect_ratio

            # Check if person has collapsed into a horizontal profile (W / H > threshold)
            is_horizontal = ar >= self.cfg.aspect_ratio_threshold

            if is_horizontal:
                state.consecutive_low_profile += 1
                if (state.consecutive_low_profile >= self.cfg.confirm_frames
                        and not state.confirmed):
                    state.confirmed = True
                    state.triggered_time = timestamp
                    events.append(FallEvent(
                        track_id=person.track_id,
                        timestamp=timestamp,
                        aspect_ratio=ar,
                        confidence=person.confidence,
                        position=person.bottom_center,
                    ))
            else:
                # Upright again: reset counter
                state.consecutive_low_profile = max(0, state.consecutive_low_profile - 1)
                if state.consecutive_low_profile == 0:
                    state.confirmed = False

        return events

    def is_fallen(self, track_id: int) -> bool:
        state = self._states.get(track_id)
        return state.confirmed if state else False
