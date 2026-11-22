"""
Weapon / Threat Detection (second YOLO model)

Runs a pretrained weapon model (gun, knife, ...) next to the normal COCO
detector. It is NOT part of the tracker: weapons are small and fast, so this
model only runs every N frames and returns plain detections. Event logic
(confirmation + alerting) lives in src/activity/weapon_alert.py.
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from src.config import WeaponConfig


@dataclass
class WeaponDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str

    @property
    def bbox_xyxy(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class WeaponDetector:
    def __init__(self, cfg: WeaponConfig):
        from ultralytics import YOLO  # lazy import so unit tests run without it

        self.cfg = cfg
        self.model = YOLO(cfg.model_path)
        self.names = {int(k): str(v).lower() for k, v in self.model.names.items()}
        watch = {w.lower() for w in cfg.watch_classes}
        self.class_ids = [i for i, n in self.names.items() if n in watch]
        print(f"[WeaponDetector] model classes: {self.names}")
        print(f"[WeaponDetector] watching: {[self.names[i] for i in self.class_ids]}")
        if not self.class_ids:
            print("[WeaponDetector] WARNING: no model class matches watch_classes "
                  "- edit WeaponConfig.watch_classes in src/config.py")

        self._frame_counter = 0
        self.ran_this_frame = False
        self.latest: List[WeaponDetection] = []   # last inference result (for drawing)
        self._latest_age = 0

    def update(self, frame: np.ndarray) -> List[WeaponDetection]:
        """Call once per frame. Runs inference only every `run_every_n_frames`.
        Returns fresh detections on inference frames, [] otherwise.
        Check `ran_this_frame` to know whether events should be evaluated."""
        self._frame_counter += 1
        self.ran_this_frame = False

        if not self.cfg.enabled or not self.class_ids:
            return []

        if (self._frame_counter - 1) % max(1, self.cfg.run_every_n_frames) != 0:
            self._latest_age += 1
            if self._latest_age > self.cfg.draw_hold_frames:
                self.latest = []
            return []

        self.ran_this_frame = True
        results = self.model.predict(
            source=frame,
            classes=self.class_ids,
            conf=self.cfg.min_confidence,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            verbose=False,
        )
        dets: List[WeaponDetection] = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cname = self.names.get(int(box.cls[0].item()), "weapon")
                conf = float(box.conf[0])
                if conf < self.cfg.class_thresholds.get(cname, self.cfg.confidence_threshold):
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append(WeaponDetection(x1, y1, x2, y2, conf, cname))

        self.latest = dets
        self._latest_age = 0
        return dets


def draw_weapons(vis: np.ndarray, detections: List[WeaponDetection]) -> np.ndarray:
    """Draw weapon boxes (thick red + label). Returns the same image."""
    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox_xyxy)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"WEAPON: {d.class_name} {d.confidence:.2f}"
        cv2.putText(vis, label, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return vis