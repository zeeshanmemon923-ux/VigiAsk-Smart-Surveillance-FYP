"""
Robustness test: run a YOLO model on a Roboflow dataset (YOLO format export) under
different conditions (dark, bright, noise, blur, fog, night, low_quality) and report
precision / recall / F1 per condition (IoU 0.5, all target classes merged).

Examples:
  # person detector on a person/CCTV dataset
  python tests/run_condition_tests.py --model yolov8n.pt --dataset data/rf_person --targets person
  # weapon model on a weapon dataset
  python tests/run_condition_tests.py --model models/weapon_yolo.pt --dataset data/rf_weapon --targets gun,pistol,knife,rifle
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.conditions import CONDITIONS, apply_condition


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match(preds, gts, thr=0.5):
    """preds: [(box, conf)], gts: [box] -> (tp, fp, fn); greedy by confidence."""
    used, tp = set(), 0
    for box, _ in sorted(preds, key=lambda p: -p[1]):
        best, bi = 0.0, -1
        for i, g in enumerate(gts):
            if i not in used and iou(box, g) > best:
                best, bi = iou(box, g), i
        if best >= thr:
            used.add(bi)
            tp += 1
    return tp, len(preds) - tp, len(gts) - tp


def ids_for(names, targets):
    names = names.items() if isinstance(names, dict) else enumerate(names)
    return [int(i) for i, n in names if str(n).lower() in targets]


def load_gt(label_file, w, h, ds_ids):
    boxes = []
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            p = line.split()
            if len(p) >= 5 and int(float(p[0])) in ds_ids:
                cx, cy, bw, bh = (float(v) for v in p[1:5])
                boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, help="Roboflow YOLO export folder (has data.yaml)")
    ap.add_argument("--split", default="test", help="test | valid | train")
    ap.add_argument("--targets", default="person", help="comma-separated class names to evaluate")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--max-images", type=int, default=150)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--out", default="data/condition_report.json")
    a = ap.parse_args()

    from ultralytics import YOLO
    root = Path(a.dataset)
    targets = {t.strip().lower() for t in a.targets.split(",")}
    ds_ids = set(ids_for(yaml.safe_load((root / "data.yaml").read_text())["names"], targets))
    model = YOLO(a.model)
    m_ids = ids_for(model.names, targets)
    if not ds_ids or not m_ids:
        sys.exit(f"No class matches {targets}. dataset ids={ds_ids}, model ids={m_ids}")

    imgs = sorted((root / a.split / "images").glob("*.*"))[: a.max_images]
    if not imgs:
        sys.exit(f"No images in {root / a.split / 'images'}")
    gts = {}
    for p in imgs:
        im = cv2.imread(str(p))
        if im is not None:
            gts[p] = (im, load_gt(root / a.split / "labels" / (p.stem + ".txt"), im.shape[1], im.shape[0], ds_ids))

    report = {}
    for cond in a.conditions.split(","):
        TP = FP = FN = 0
        for p, (im, gt) in gts.items():
            r = model.predict(apply_condition(im, cond), conf=a.conf, classes=m_ids, verbose=False)[0]
            preds = [(b.xyxy[0].tolist(), float(b.conf[0])) for b in r.boxes]
            tp, fp, fn = match(preds, gt)
            TP, FP, FN = TP + tp, FP + fp, FN + fn
        P = TP / (TP + FP) if TP + FP else 0.0
        R = TP / (TP + FN) if TP + FN else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        report[cond] = {"precision": round(P, 3), "recall": round(R, 3), "f1": round(F, 3), "tp": TP, "fp": FP, "fn": FN}
        print(f"{cond:12s} P={P:.3f} R={R:.3f} F1={F:.3f}  (TP={TP} FP={FP} FN={FN})")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"model": a.model, "dataset": str(root), "images": len(gts), "results": report}, indent=2))
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
