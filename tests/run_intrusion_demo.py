"""
Step 3 sanity check: tracking -> restricted-area intrusion detection.

Draw a zone first with tests/zone_editor.py, then run this — a person
whose foot-point stays inside the zone for a few frames triggers a
red flash + "INTRUSION" label, and the event gets appended to
data/events.jsonl.

Usage:
    python tests/zone_editor.py --source data/sample.mp4      # draw zone once
    python tests/run_intrusion_demo.py --source data/sample.mp4
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG
from src.utils.preprocessing import VideoSource
from src.utils.event_logger import EventLogger
from src.tracking.tracker import PersonTracker
from src.activity.intrusion import IntrusionDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    tracker = PersonTracker(CONFIG.detection, CONFIG.tracking)
    intrusion = IntrusionDetector(CONFIG.intrusion)
    logger = EventLogger(CONFIG.events_log_path)

    if not intrusion.zones:
        print(f"No zones found in {CONFIG.intrusion.zones_file}. "
              f"Run tests/zone_editor.py first to draw one.")

    with VideoSource(source, CONFIG.preprocess) as video:
        for frame in video:
            people = tracker.update(frame.image, frame.timestamp_sec)
            events = intrusion.update(people, frame.timestamp_sec)

            for ev in events:
                print(f"[INTRUSION] track {ev.track_id} entered '{ev.zone_name}' "
                      f"at t={ev.timestamp:.2f}s")
                logger.log(ev.to_dict())

            if not args.no_display:
                vis = frame.image.copy()

                # draw zones
                for zone in intrusion.zones:
                    pts = zone.polygon
                    for i in range(len(pts)):
                        p1 = tuple(map(int, pts[i]))
                        p2 = tuple(map(int, pts[(i + 1) % len(pts)]))
                        cv2.line(vis, p1, p2, (0, 0, 255), 2)
                    cv2.putText(vis, zone.name, tuple(map(int, pts[0])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                inside_now = {tid for tid, _ in intrusion.currently_inside()}

                for p in people:
                    is_intruding = p.track_id in inside_now
                    color = (0, 0, 255) if is_intruding else (0, 255, 0)
                    x1, y1, x2, y2 = map(int, p.bbox_xyxy)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    label = f"ID {p.track_id}" + (" INTRUSION" if is_intruding else "")
                    cv2.putText(vis, label, (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                cv2.imshow("Restricted-Area Intrusion - Step 3", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
