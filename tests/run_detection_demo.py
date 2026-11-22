"""
Step 1 sanity check: preprocessing -> person detection, on a video file
or webcam, with boxes drawn live so you can SEE it's working before
tracking/activity logic gets layered on top.

Usage:
    python tests/run_detection_demo.py --source data/sample.mp4
    python tests/run_detection_demo.py --source 0          # webcam
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG
from src.utils.preprocessing import VideoSource
from src.detection.person_detector import PersonDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="video file path or webcam index")
    parser.add_argument("--no-display", action="store_true", help="run headless, just print FPS")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    detector = PersonDetector(CONFIG.detection)

    with VideoSource(source, CONFIG.preprocess) as video:
        frame_times = []
        for frame in video:
            t0 = time.time()
            detections = detector.detect(frame.image)
            elapsed = time.time() - t0
            frame_times.append(elapsed)

            print(
                f"frame {frame.index:5d} | t={frame.timestamp_sec:6.2f}s | "
                f"people={len(detections):2d} | latency={elapsed*1000:.1f}ms"
            )

            if not args.no_display:
                vis = frame.image.copy()
                for det in detections:
                    x1, y1, x2, y2 = map(int, det.bbox_xyxy)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        vis, f"{det.confidence:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    )
                cv2.imshow("Person Detection - Step 1", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if frame_times:
            avg_latency = sum(frame_times) / len(frame_times)
            print(f"\nAvg detection latency: {avg_latency*1000:.1f} ms "
                  f"({1/avg_latency:.1f} FPS)")

    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
