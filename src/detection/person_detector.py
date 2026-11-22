"""
Person Detection (Proposal Section 4.2)

Uses a pretrained YOLOv8 model (ultralytics) to detect people in each
video frame, filtered to the COCO "person" class. Returns bounding
boxes + confidence scores that feed straight into the tracker.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from src.config import DetectionConfig


@dataclass
class Detection:
    """One detected object in one frame."""
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
        """Foot point — better than centroid for ground-plane zone checks."""
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


class ObjectDetector:
    """
    General YOLOv8 object detector supporting multiple classes (e.g. Person,
    Backpack, Handbag, Suitcase) with per-class confidence thresholds.
    """

    def __init__(self, cfg: DetectionConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)
        # Fallback class map from model if not in config
        self.names = getattr(self.model, "names", {})

    def detect(self, frame: np.ndarray, classes: Optional[List[int]] = None) -> List[Detection]:
        """Run detection on a single BGR frame, returning filtered detections."""
        target_classes = classes if classes is not None else getattr(self.cfg, "target_classes", [self.cfg.person_class_id])
        results = self.model.predict(
            source=frame,
            classes=target_classes,
            conf=self.cfg.confidence_threshold,
            iou=self.cfg.iou_threshold,
            device=self.cfg.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        class_thresholds = getattr(self.cfg, "class_thresholds", {})

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cid = int(box.cls[0].item()) if hasattr(box, "cls") and box.cls is not None else 0
            cname = self.names.get(cid, "person")

            min_conf = class_thresholds.get(cname, self.cfg.confidence_threshold)
            if conf >= min_conf:
                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf,
                    class_id=cid,
                    class_name=cname
                ))

        return detections

    def detect_batch(self, frames: List[np.ndarray], classes: Optional[List[int]] = None) -> List[List[Detection]]:
        """Batch inference for throughput when processing recorded footage."""
        target_classes = classes if classes is not None else getattr(self.cfg, "target_classes", [self.cfg.person_class_id])
        results = self.model.predict(
            source=frames,
            classes=target_classes,
            conf=self.cfg.confidence_threshold,
            iou=self.cfg.iou_threshold,
            device=self.cfg.device,
            verbose=False,
        )
        all_detections: List[List[Detection]] = []
        class_thresholds = getattr(self.cfg, "class_thresholds", {})

        for r in results:
            frame_dets = []
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cid = int(box.cls[0].item()) if hasattr(box, "cls") and box.cls is not None else 0
                    cname = self.names.get(cid, "person")
                    min_conf = class_thresholds.get(cname, self.cfg.confidence_threshold)
                    if conf >= min_conf:
                        frame_dets.append(Detection(
                            x1=x1, y1=y1, x2=x2, y2=y2,
                            confidence=conf,
                            class_id=cid,
                            class_name=cname
                        ))
            all_detections.append(frame_dets)
        return all_detections


class PersonDetector(ObjectDetector):
    """
    Subclass preserving backward compatibility for existing code and tests,
    restricting detections by default to the person class (COCO 0).
    """

    def detect(self, frame: np.ndarray, classes: Optional[List[int]] = None) -> List[Detection]:
        target = [self.cfg.person_class_id] if classes is None else classes
        return super().detect(frame, classes=target)

