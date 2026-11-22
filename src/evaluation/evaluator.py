"""
Detector Evaluation (Objective 3 / Milestone 11)

"...analyse the proposed system using suitable performance measures
like precision, recall, F1-score, false-positive rate, detection
latency, and processing speed" — this module computes exactly those,
against the ground-truth set built with tests/ground_truth_annotator.py.

Definitions used (standard for object detection):
  precision      = TP / (TP + FP)            -- of what we flagged, how much was real
  recall         = TP / (TP + FN)             -- of what was real, how much we caught
  F1-score       = 2 * precision * recall / (precision + recall)
  FP rate        = FP / (TP + FP)             -- fraction of detections that were wrong
                   (also reported: FP per labeled frame, a second common framing)
  latency        = seconds per detect() call, measured directly
  processing speed = 1 / mean_latency, in FPS
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from src.config import EvalConfig, PreprocessConfig
from src.detection.person_detector import PersonDetector
from src.evaluation.metrics import match_detections
from src.utils.preprocessing import VideoSource


@dataclass
class EvaluationReport:
    num_frames_evaluated: int
    total_ground_truth: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    fp_rate: float
    fp_per_frame: float
    mean_latency_ms: float
    fps: float

    def to_dict(self):
        return self.__dict__

    def print_summary(self):
        print("\n=== Evaluation Report ===")
        print(f"Frames evaluated:      {self.num_frames_evaluated}")
        print(f"Ground-truth people:   {self.total_ground_truth}")
        print(f"True Positives:        {self.true_positives}")
        print(f"False Positives:       {self.false_positives}")
        print(f"False Negatives:       {self.false_negatives}")
        print(f"Precision:             {self.precision:.3f}")
        print(f"Recall:                {self.recall:.3f}")
        print(f"F1-score:              {self.f1_score:.3f}")
        print(f"False-Positive Rate:   {self.fp_rate:.3f}")
        print(f"FP per labeled frame:  {self.fp_per_frame:.3f}")
        print(f"Mean latency:          {self.mean_latency_ms:.1f} ms")
        print(f"Processing speed:      {self.fps:.1f} FPS")


def load_ground_truth(path: str) -> Dict[str, List[List[float]]]:
    with open(path) as f:
        data = json.load(f)
    return data.get("frames", {})


def evaluate(
    source,
    ground_truth_path: str,
    detector: PersonDetector,
    preprocess_cfg: PreprocessConfig,
    eval_cfg: EvalConfig,
) -> EvaluationReport:
    gt_frames = load_ground_truth(ground_truth_path)
    if not gt_frames:
        raise ValueError(
            f"No labeled frames in {ground_truth_path}. "
            f"Run tests/ground_truth_annotator.py first."
        )

    total_tp = total_fp = total_fn = 0
    total_gt = 0
    latencies: List[float] = []
    frames_checked = 0
    warmed_up = False

    with VideoSource(source, preprocess_cfg) as video:
        for frame in video:
            if not warmed_up:
                # first inference call loads weights / builds the graph —
                # a one-time cost that would otherwise dominate the average
                # latency on a small evaluation set. Run it once, untimed.
                detector.detect(frame.image)
                warmed_up = True

            key = str(frame.index)
            if key not in gt_frames:
                continue

            gt_boxes = [tuple(b) for b in gt_frames[key]]

            t0 = time.time()
            detections = detector.detect(frame.image)
            latencies.append(time.time() - t0)

            preds = [(d.bbox_xyxy, d.confidence) for d in detections]
            tp, fp, fn = match_detections(preds, gt_boxes, eval_cfg.iou_match_threshold)

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_gt += len(gt_boxes)
            frames_checked += 1

    if frames_checked == 0:
        raise ValueError(
            "None of the ground-truth frame indices were found in this video. "
            "Make sure --source matches the video used for annotation."
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fp_rate = total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    fp_per_frame = total_fp / frames_checked

    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return EvaluationReport(
        num_frames_evaluated=frames_checked,
        total_ground_truth=total_gt,
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        fp_rate=fp_rate,
        fp_per_frame=fp_per_frame,
        mean_latency_ms=mean_latency * 1000,
        fps=(1 / mean_latency) if mean_latency > 0 else 0.0,
    )
