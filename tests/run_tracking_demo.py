"""
Step 2 sanity check: detection -> ByteTrack tracking, on a video file
or webcam. Each person gets a stable ID and a trail of recent
positions, so you can visually confirm the same person keeps the same
ID as they move (and that IDs don't jump/swap on occlusion).

Usage:
    python tests/run_tracking_demo.py --source data/sample.mp4
    python tests/run_tracking_demo.py --source 0
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG
from src.utils.preprocessing import VideoSource
from src.tracking.tracker import PersonTracker


# distinct-ish colors per track id, cycled
COLORS = [
    (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255),
    (0, 255, 255), (128, 0, 255), (255, 255, 0), (0, 0, 255),
]


def color_for(track_id: int):
    return COLORS[track_id % len(COLORS)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="video file path or webcam index")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    tracker = PersonTracker(CONFIG.detection, CONFIG.tracking)

    with VideoSource(source, CONFIG.preprocess) as video:
        frame_times = []
        for frame in video:
            t0 = time.time()
            people = tracker.update(frame.image, frame.timestamp_sec)
            elapsed = time.time() - t0
            frame_times.append(elapsed)

            ids = [p.track_id for p in people]
            print(
                f"frame {frame.index:5d} | t={frame.timestamp_sec:6.2f}s | "
                f"tracks={ids} | latency={elapsed*1000:.1f}ms"
            )

            if not args.no_display:
                vis = frame.image.copy()
                for p in people:
                    color = color_for(p.track_id)
                    x1, y1, x2, y2 = map(int, p.bbox_xyxy)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        vis, f"ID {p.track_id} ({p.confidence:.2f})",
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                    )
                    # motion trail from tracking history
                    hist = tracker.get_history(p.track_id)
                    if hist and len(hist.positions) > 1:
                        pts = [(int(x), int(y)) for x, y in hist.positions[-30:]]
                        for i in range(1, len(pts)):
                            cv2.line(vis, pts[i - 1], pts[i], color, 2)

                    # how long this ID has been tracked, useful preview of loitering logic
                    if hist:
                        cv2.putText(
                            vis, f"{hist.duration_seconds:.1f}s",
                            (x1, min(vis.shape[0] - 5, int(y2) + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        )

                cv2.imshow("Person Tracking - Step 2", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            tracker.prune_stale(frame.timestamp_sec)

        if frame_times:
            avg_latency = sum(frame_times) / len(frame_times)
            print(f"\nAvg tracking latency: {avg_latency*1000:.1f} ms "
                  f"({1/avg_latency:.1f} FPS)")

    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
