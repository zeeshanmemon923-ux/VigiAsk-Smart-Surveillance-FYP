"""
Ground-Truth Annotator — for building the small labeled test set the
proposal's Milestone 4 calls for ("prepare a small custom CCTV dataset
for testing and evaluation").

Samples every Nth frame from a video, lets you draw a box around each
person by clicking two corners, and saves everything to a JSON file
that src/evaluation/evaluator.py reads.

Usage:
    python tests/ground_truth_annotator.py --source data/sample.mp4 --every 30

Controls (per sampled frame):
    click, click     draw one box (top-left corner, then bottom-right corner)
    z                undo last box on this frame
    n / space        confirm this frame's boxes, move to next sampled frame
    s                skip this frame (no people / don't want to label it)
    q                save everything labeled so far, and quit
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG
from src.utils.preprocessing import VideoSource


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--every", type=int, default=30,
                         help="label every Nth frame (default: 30, ~1/sec at 30fps)")
    parser.add_argument("--out", default="data/ground_truth.json")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    labels = {}   # frame_index (str) -> list of [x1, y1, x2, y2]
    pending_points = []   # up to 2 points for the box currently being drawn
    current_boxes = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_points.append((x, y))
            if len(pending_points) == 2:
                (x1, y1), (x2, y2) = pending_points
                box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                current_boxes.append(box)
                pending_points.clear()

    win = "GT Annotator - click 2 corners per person, 'n' next, 's' skip, 'q' save+quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)

    print(f"Sampling every {args.every} frames. For each frame shown: click two "
          f"corners per person, 'n'/space to confirm and advance, 's' to skip, "
          f"'z' to undo last box, 'q' to save and quit.")

    with VideoSource(source, CONFIG.preprocess) as video:
        for frame in video:
            if frame.index % args.every != 0:
                continue

            current_boxes = []
            pending_points.clear()

            while True:
                vis = frame.image.copy()
                for b in current_boxes:
                    cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                for p in pending_points:
                    cv2.circle(vis, p, 4, (0, 255, 255), -1)
                cv2.putText(vis, f"frame {frame.index}  boxes: {len(current_boxes)}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow(win, vis)

                key = cv2.waitKey(20) & 0xFF
                if key in (ord("n"), ord(" ")):
                    labels[str(frame.index)] = current_boxes
                    break
                elif key == ord("s"):
                    labels[str(frame.index)] = []  # confirmed empty (no people)
                    break
                elif key == ord("z") and current_boxes:
                    current_boxes.pop()
                elif key == ord("q"):
                    _save(labels, args.out)
                    cv2.destroyAllWindows()
                    return

    _save(labels, args.out)
    cv2.destroyAllWindows()


def _save(labels: dict, out_path: str):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"frames": labels}, f, indent=2)
    total_boxes = sum(len(v) for v in labels.values())
    print(f"Saved {len(labels)} labeled frame(s), {total_boxes} box(es) total, to {path}")


if __name__ == "__main__":
    main()
