"""
Zone Editor — click points on a frame to define a restricted-area
polygon, save it to data/zones.json. Run this once per camera/scene
before using intrusion detection.

Usage:
    python tests/zone_editor.py --source data/sample.mp4
    python tests/zone_editor.py --source 0

Controls:
    left-click   add a point to the current polygon
    n            finish current polygon, start a new one
    s            save all polygons to data/zones.json and exit
    z            undo last point
    q            quit without saving
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
    parser.add_argument("--out", default=CONFIG.intrusion.zones_file)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    # grab a single representative frame to draw on
    with VideoSource(source, CONFIG.preprocess) as video:
        frame = next(iter(video))
        still = frame.image.copy()

    zones = []
    current = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))

    win = "Zone Editor - left-click points, 'n' next zone, 's' save, 'q' quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)

    print("Click to add points. Press 'n' to close a polygon and start the "
          "next, 's' to save, 'z' to undo a point, 'q' to quit without saving.")

    zone_id = 1
    while True:
        vis = still.copy()

        # draw already-finished zones
        for z in zones:
            pts = z["polygon"]
            for i in range(len(pts)):
                cv2.line(vis, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 0, 255), 2)
            cv2.putText(vis, z["name"], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

        # draw current in-progress polygon
        for i, p in enumerate(current):
            cv2.circle(vis, p, 4, (0, 255, 255), -1)
            if i > 0:
                cv2.line(vis, current[i - 1], p, (0, 255, 255), 2)

        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("n") and len(current) >= 3:
            zones.append({"name": f"zone_{zone_id}", "polygon": [list(p) for p in current]})
            zone_id += 1
            current = []
        elif key == ord("z") and current:
            current.pop()
        elif key == ord("s"):
            if len(current) >= 3:
                zones.append({"name": f"zone_{zone_id}", "polygon": [list(p) for p in current]})
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({"zones": zones}, f, indent=2)
            print(f"Saved {len(zones)} zone(s) to {out_path}")
            break
        elif key == ord("q"):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
