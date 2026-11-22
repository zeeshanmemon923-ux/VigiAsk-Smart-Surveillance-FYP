"""
Observable High-Energy Interaction / Altercation Module

Detects close physical proximity between multiple tracked persons combined
with high-variance, rapid spatial motion over consecutive frames.
Adheres strictly to objective behavioral criteria: describes observable
motion events, never assuming criminal culpability.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Set, Tuple

from src.config import AltercationConfig
from src.tracking.tracker import TrackedPerson, TrackHistory


@dataclass
class AltercationEvent:
    track_ids: List[int]
    timestamp: float
    confidence: float
    position: Tuple[float, float]

    def to_dict(self):
        return {
            "event_type": "high_energy_interaction",
            "track_ids": self.track_ids,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "position": list(self.position),
        }


@dataclass
class _PairInteractionState:
    consecutive_frames: int = 0
    confirmed: bool = False
    last_seen: float = 0.0


class AltercationDetector:
    """
    Stateful detector analyzing interaction between pairs of people.
    Flags sudden rapid physical movement while in close contact.
    """

    def __init__(self, cfg: AltercationConfig):
        self.cfg = cfg
        self._pair_states: Dict[Tuple[int, int], _PairInteractionState] = {}

    def update(
        self,
        people: List[TrackedPerson],
        histories: Dict[int, TrackHistory],
        timestamp: float,
    ) -> List[AltercationEvent]:
        events: List[AltercationEvent] = []
        if not self.cfg.enabled or len(people) < 2:
            return events

        # Consider only person tracks
        persons = [p for p in people if p.class_name == "person"]
        if len(persons) < 2:
            return events

        active_pairs: Set[Tuple[int, int]] = set()

        for p1, p2 in combinations(persons, 2):
            pair_key = (min(p1.track_id, p2.track_id), max(p1.track_id, p2.track_id))
            active_pairs.add(pair_key)

            pos1 = p1.bottom_center
            pos2 = p2.bottom_center
            dist = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

            in_proximity = dist <= self.cfg.proximity_threshold_px

            # Check velocity variance / rapid displacement
            h1 = histories.get(p1.track_id)
            h2 = histories.get(p2.track_id)

            rapid_motion = False
            if h1 and h2 and len(h1.positions) >= 2 and len(h2.positions) >= 2:
                # Calculate displacement magnitude over last frame
                disp1 = (
                    (h1.positions[-1][0] - h1.positions[-2][0]) ** 2 +
                    (h1.positions[-1][1] - h1.positions[-2][1]) ** 2
                ) ** 0.5
                disp2 = (
                    (h2.positions[-1][0] - h2.positions[-2][0]) ** 2 +
                    (h2.positions[-1][1] - h2.positions[-2][1]) ** 2
                ) ** 0.5

                if (disp1 + disp2) >= self.cfg.velocity_variance_threshold:
                    rapid_motion = True

            state = self._pair_states.setdefault(pair_key, _PairInteractionState())
            state.last_seen = timestamp

            if in_proximity and rapid_motion:
                state.consecutive_frames += 1
                if (state.consecutive_frames >= self.cfg.confirm_frames
                        and not state.confirmed):
                    state.confirmed = True
                    midpoint = ((pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2)
                    avg_conf = (p1.confidence + p2.confidence) / 2
                    events.append(AltercationEvent(
                        track_ids=list(pair_key),
                        timestamp=timestamp,
                        confidence=avg_conf,
                        position=midpoint,
                    ))
            else:
                state.consecutive_frames = max(0, state.consecutive_frames - 1)
                if state.consecutive_frames == 0:
                    state.confirmed = False

        # Prune dead pairs
        for pair_key in list(self._pair_states.keys()):
            if pair_key not in active_pairs or (timestamp - self._pair_states[pair_key].last_seen) > 3.0:
                del self._pair_states[pair_key]

        return events

    def is_in_interaction(self, track_id: int) -> bool:
        for (id1, id2), state in self._pair_states.items():
            if (track_id == id1 or track_id == id2) and state.confirmed:
                return True
        return False
