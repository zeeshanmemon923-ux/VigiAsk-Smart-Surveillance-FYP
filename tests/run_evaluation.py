"""
Step 6: run evaluation against your labeled ground-truth set and
produce a report you can drop straight into the thesis (Objective 3 /
Milestone 11: precision, recall, F1-score, false-positive rate,
detection latency, processing speed).

Usage:
    python tests/ground_truth_annotator.py --source data/sample.mp4  # once
    python tests/run_evaluation.py --source data/sample.mp4
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG
from src.detection.person_detector import PersonDetector
from src.evaluation.evaluator import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/sample.mp4")
    parser.add_argument("--ground-truth", default="data/ground_truth.json")
    parser.add_argument("--out", default="data/evaluation_report.json")
    parser.add_argument("--chart", default="data/evaluation_chart.png")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    detector = PersonDetector(CONFIG.detection)
    report = evaluate(
        source=source,
        ground_truth_path=args.ground_truth,
        detector=detector,
        preprocess_cfg=CONFIG.preprocess,
        eval_cfg=CONFIG.evaluation,
    )

    report.print_summary()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nSaved full report to {out_path}")

    _save_chart(report, args.chart)
    print(f"Saved chart to {args.chart}")


def _save_chart(report, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    metrics = ["Precision", "Recall", "F1-score", "FP rate"]
    values = [report.precision, report.recall, report.f1_score, report.fp_rate]
    axes[0].bar(metrics, values, color=["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Detection Quality")
    for i, v in enumerate(values):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center")

    axes[1].bar(["Latency (ms)", "FPS"],
                [report.mean_latency_ms, report.fps],
                color=["#f39c12", "#1abc9c"])
    axes[1].set_title("Speed")
    for i, v in enumerate([report.mean_latency_ms, report.fps]):
        axes[1].text(i, v, f"{v:.1f}", ha="center", va="bottom")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)


if __name__ == "__main__":
    main()
