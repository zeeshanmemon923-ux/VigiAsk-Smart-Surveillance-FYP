"""
Detection-matching helpers for evaluation.

Standard object-detection evaluation: match predicted boxes to
ground-truth boxes by IoU (Intersection-over-Union), greedily, highest
confidence first. A prediction only counts as a True Positive if its
IoU with an unmatched ground-truth box clears the threshold.
"""

from typing import List, Tuple


def iou(box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float]) -> float:
    """IoU between two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def match_detections(
    predictions: List[Tuple[Tuple[float, float, float, float], float]],
    ground_truths: List[Tuple[float, float, float, float]],
    iou_threshold: float,
):
    """
    predictions: list of (box, confidence), any order.
    ground_truths: list of boxes.

    Returns (true_positives, false_positives, false_negatives) as counts,
    matched greedily by descending confidence.
    """
    preds_sorted = sorted(predictions, key=lambda p: p[1], reverse=True)
    matched_gt = set()
    tp = 0
    fp = 0

    for box, _conf in preds_sorted:
        best_iou = 0.0
        best_idx = -1
        for i, gt_box in enumerate(ground_truths):
            if i in matched_gt:
                continue
            score = iou(box, gt_box)
            if score > best_iou:
                best_iou = score
                best_idx = i

        if best_iou >= iou_threshold and best_idx != -1:
            matched_gt.add(best_idx)
            tp += 1
        else:
            fp += 1

    fn = len(ground_truths) - len(matched_gt)
    return tp, fp, fn
