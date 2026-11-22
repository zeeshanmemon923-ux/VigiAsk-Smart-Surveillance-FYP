"""
Multi-Object Tracking (Proposal Section 4.3)

Assigns a persistent tracking ID to each detected person and maintains
that identity across consecutive frames. This is what makes "how long
has this person been in this spot" (loitering) and "did this person
cross into the zone" (intrusion) answerable — detection alone only
knows about a single frame at a time.

Uses Ultralytics' built-in ByteTrack (bytetrack.yaml) by default, since
YOLO + ByteTrack are already in the same library and it keeps the
dependency footprint small. DeepSORT is kept as a swappable alternative
(TrackingConfig.tracker = "deepsort") since the proposal names both.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from ultralytics import YOLO

from src.config import TrackingConfig, DetectionConfig


@dataclass
class TrackedPerson:
    """One tracked object (person or item) in one frame."""
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0
    class_name: str = "person"

    @property
    def bbox_xyxy(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def bottom_center(self):
        """Foot point — used for ground-plane zone/loitering checks."""
        return ((self.x1 + self.x2) / 2, self.y2)

    @property
    def width(self):
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self):
        return max(0.0, self.y2 - self.y1)

    @property
    def aspect_ratio(self):
        """Width / Height ratio — useful for posture/fall detection."""
        return self.width / self.height if self.height > 0 else 0.0


# Alias for general object tracking
TrackedObject = TrackedPerson


@dataclass
class TrackHistory:
    """Rolling history for one track ID — positions, bboxes, timestamps."""
    track_id: int
    positions: List[tuple] = field(default_factory=list)   # (x, y) bottom-center
    bboxes: List[tuple] = field(default_factory=list)      # (x1, y1, x2, y2)
    timestamps: List[float] = field(default_factory=list)  # seconds
    class_name: str = "person"
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None

    def update(self, position, bbox, timestamp: float, max_len: int = 300, class_name: str = "person"):
        if self.first_seen is None:
            self.first_seen = timestamp
        self.last_seen = timestamp
        self.class_name = class_name
        self.positions.append(position)
        self.bboxes.append(bbox)
        self.timestamps.append(timestamp)
        if len(self.positions) > max_len:
            self.positions.pop(0)
            self.bboxes.pop(0)
            self.timestamps.pop(0)

    @property
    def duration_seconds(self) -> float:
        if self.first_seen is None or self.last_seen is None:
            return 0.0
        return self.last_seen - self.first_seen

    @property
    def current_aspect_ratio(self) -> float:
        if not self.bboxes:
            return 0.0
        x1, y1, x2, y2 = self.bboxes[-1]
        h = max(1.0, y2 - y1)
        w = max(0.0, x2 - x1)
        return w / h

    @property
    def vertical_velocity(self) -> float:
        """Rate of downward centroid/foot displacement (pixels per second)."""
        if len(self.positions) < 2 or len(self.timestamps) < 2:
            return 0.0
        dt = self.timestamps[-1] - self.timestamps[-2]
        if dt <= 0:
            return 0.0
        dy = self.positions[-1][1] - self.positions[-2][1]
        return dy / dt


class PersonTracker:
    """
    Wraps YOLOv8's built-in tracking (ByteTrack) and DeepSORT.
    Maintains per-track history for downstream activity detectors
    (intrusion, loitering, unattended objects, falls, altercation indicators).
    """

    def __init__(self, det_cfg: DetectionConfig, track_cfg: TrackingConfig):
        self.det_cfg = det_cfg
        self.track_cfg = track_cfg
        self.model = YOLO(det_cfg.model_path)
        self.names = getattr(self.model, "names", {})
        self.histories: Dict[int, TrackHistory] = {}

        if track_cfg.tracker == "deepsort":
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self._deepsort = DeepSort(max_age=track_cfg.track_buffer)
        else:
            self._deepsort = None

    def update(self, frame: np.ndarray, timestamp: float, classes: Optional[List[int]] = None) -> List[TrackedPerson]:
        """Run detection+tracking on one frame, update history, return current tracks."""
        if self.track_cfg.tracker == "deepsort":
            people = self._update_deepsort(frame, classes=classes)
        else:
            people = self._update_bytetrack(frame, classes=classes)

        for p in people:
            hist = self.histories.setdefault(p.track_id, TrackHistory(track_id=p.track_id))
            hist.update(p.bottom_center, p.bbox_xyxy, timestamp, class_name=p.class_name)

        return people

    def update_all(self, frame: np.ndarray, timestamp: float) -> List[TrackedPerson]:
        """Track all configured target classes (e.g. people and luggage/bags)."""
        target_classes = getattr(self.det_cfg, "target_classes", [self.det_cfg.person_class_id])
        return self.update(frame, timestamp, classes=target_classes)

    def _update_bytetrack(self, frame: np.ndarray, classes: Optional[List[int]] = None) -> List[TrackedPerson]:
        target_classes = [self.det_cfg.person_class_id] if classes is None else classes
        results = self.model.track(
            source=frame,
            classes=target_classes,
            conf=self.det_cfg.confidence_threshold,
            iou=self.det_cfg.iou_threshold,
            device=self.det_cfg.device,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )

        people: List[TrackedPerson] = []
        if not results or results[0].boxes is None:
            return people

        boxes = results[0].boxes
        if boxes.id is None:
            return people

        for box, tid in zip(boxes, boxes.id):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cid = int(box.cls[0].item()) if hasattr(box, "cls") and box.cls is not None else 0
            cname = self.names.get(cid, "person")
            people.append(TrackedPerson(
                track_id=int(tid.item()),
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=conf,
                class_id=cid,
                class_name=cname
            ))

        return people

    def _update_deepsort(self, frame: np.ndarray, classes: Optional[List[int]] = None) -> List[TrackedPerson]:
        target_classes = [self.det_cfg.person_class_id] if classes is None else classes
        det_results = self.model.predict(
            source=frame,
            classes=target_classes,
            conf=self.det_cfg.confidence_threshold,
            iou=self.det_cfg.iou_threshold,
            device=self.det_cfg.device,
            verbose=False,
        )
        raw_dets = []
        if det_results and det_results[0].boxes is not None:
            for box in det_results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cid = int(box.cls[0].item()) if hasattr(box, "cls") and box.cls is not None else 0
                cname = self.names.get(cid, "person")
                raw_dets.append(([x1, y1, x2 - x1, y2 - y1], conf, cname))

        tracks = self._deepsort.update_tracks(raw_dets, frame=frame)

        people: List[TrackedPerson] = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            x1, y1, x2, y2 = t.to_ltrb()
            cname = t.get_det_class() if hasattr(t, "get_det_class") else "person"
            people.append(TrackedPerson(
                track_id=int(t.track_id),
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=1.0,
                class_id=0 if cname == "person" else 1,
                class_name=cname or "person"
            ))
        return people

    def get_history(self, track_id: int) -> Optional[TrackHistory]:
        return self.histories.get(track_id)

    def prune_stale(self, current_timestamp: float, max_age_seconds: float = 5.0):
        """Drop histories for tracks not seen recently, so memory doesn't grow forever."""
        stale = [
            tid for tid, h in self.histories.items()
            if h.last_seen is not None and (current_timestamp - h.last_seen) > max_age_seconds
        ]
        for tid in stale:
            del self.histories[tid]
