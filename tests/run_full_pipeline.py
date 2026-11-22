"""
Step 4 — Full pipeline demo: tracking + restricted-area intrusion +
loitering, all running together. This is the actual deliverable for
your part of the FYP — everything before this was building blocks.

Usage:
    python tests/zone_editor.py --source 0        # draw a zone once, if not done
    python tests/run_full_pipeline.py --source 0
"""

import argparse
import sys
from pathlib import Path
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"
import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG
from src.utils.preprocessing import VideoSource
from src.tracking.tracker import PersonTracker
from src.activity.intrusion import IntrusionDetector
from src.activity.loitering import LoiteringDetector
from src.activity.suspicious_object import SuspiciousObjectDetector
from src.activity.fall_detection import FallDetector
from src.activity.altercation import AltercationDetector
from src.alert.alert_manager import AlertManager
from src.detection.weapon_detector import WeaponDetector, draw_weapons
from src.activity.weapon_alert import WeaponEventDetector
from src.database.db_manager import DatabaseManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    tracker = PersonTracker(CONFIG.detection, CONFIG.tracking)
    intrusion = IntrusionDetector(CONFIG.intrusion)
    loitering = LoiteringDetector(CONFIG.loitering)
    unattended = SuspiciousObjectDetector(CONFIG.suspicious_object)
    fall = FallDetector(CONFIG.fall_detection)
    altercation = AltercationDetector(CONFIG.altercation)
    weapon_det = WeaponDetector(CONFIG.weapon)
    weapon_events = WeaponEventDetector(CONFIG.weapon)

    db = DatabaseManager(CONFIG.database.db_path)
    alert_mgr = AlertManager(CONFIG.alert, db, CONFIG.events_log_path)

    print(f"Starting Smart CCTV Full Pipeline on source: {source}")
    print(f"Database: {CONFIG.database.db_path} | Zones loaded: {len(intrusion.zones)}")

    with VideoSource(source, CONFIG.preprocess) as video:
        for frame in video:
            # Track configured target classes
            tracked_items = tracker.update(
                frame.image,
                frame.timestamp_sec,
                classes=CONFIG.detection.target_classes
            )

            people = [item for item in tracked_items if item.class_name == "person"]
            objects = [item for item in tracked_items if item.class_name != "person"]

            # 1. Intrusion Detection
            intrusion_events = intrusion.update(people, frame.timestamp_sec)
            for ev in intrusion_events:
                al = alert_mgr.process_event(ev.to_dict(), str(source), frame.image)
                if al:
                    print(f"[INTRUSION ALERT] Track #{ev.track_id} entered '{ev.zone_name}' at t={ev.timestamp:.2f}s")

            # 2. Loitering Detection
            loitering_events = loitering.update(people, tracker.histories, frame.timestamp_sec)
            for ev in loitering_events:
                al = alert_mgr.process_event(ev.to_dict(), str(source), frame.image)
                if al:
                    print(f"[LOITERING ALERT] Track #{ev.track_id} lingered {ev.duration_seconds:.1f}s at t={ev.timestamp:.2f}s")

            # 3. Suspicious / Unattended Object Detection
            unattended_events = unattended.update(objects, people, frame.timestamp_sec)
            for ev in unattended_events:
                al = alert_mgr.process_event(ev.to_dict(), str(source), frame.image)
                if al:
                    print(f"[UNATTENDED OBJECT] {ev.object_type} #{ev.track_id} left for {ev.duration_seconds:.1f}s at t={ev.timestamp:.2f}s")

            # 4. Fall Detection
            fall_events = fall.update(people, tracker.histories, frame.timestamp_sec)
            for ev in fall_events:
                al = alert_mgr.process_event(ev.to_dict(), str(source), frame.image)
                if al:
                    print(f"[POSSIBLE FALL] Track #{ev.track_id} posture collapse (AR: {ev.aspect_ratio:.2f}) at t={ev.timestamp:.2f}s")

            # 5. Altercation / Rapid Interaction
            altercation_events = altercation.update(people, tracker.histories, frame.timestamp_sec)
            for ev in altercation_events:
                al = alert_mgr.process_event(ev.to_dict(), str(source), frame.image)
                if al:
                    print(f"[HIGH-ENERGY INTERACTION] Tracks {ev.track_ids} rapid motion at t={ev.timestamp:.2f}s")

            # 6. Weapon Detection (second YOLO model, runs every N frames)
            weapon_dets = weapon_det.update(frame.image)
            if weapon_det.ran_this_frame:
                for ev in weapon_events.update(weapon_dets, people, frame.timestamp_sec):
                    evidence = draw_weapons(frame.image.copy(), weapon_dets)
                    al = alert_mgr.process_event(ev.to_dict(), str(source), evidence)
                    if al:
                        print(f"[WEAPON ALERT] {ev.weapon_class} ({ev.confidence:.2f}) holder=#{ev.holder_track_id} at t={ev.timestamp:.2f}s")

            if not args.no_display:
                vis = frame.image.copy()

                for zone in intrusion.zones:
                    pts = zone.polygon
                    for i in range(len(pts)):
                        p1 = tuple(map(int, pts[i]))
                        p2 = tuple(map(int, pts[(i + 1) % len(pts)]))
                        cv2.line(vis, p1, p2, (0, 0, 255), 2)
                    cv2.putText(vis, zone.name, tuple(map(int, pts[0])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                inside_now = {tid for tid, _ in intrusion.currently_inside()}

                for item in tracked_items:
                    is_intruding = item.track_id in inside_now
                    is_loitering = loitering.is_loitering(item.track_id)
                    is_unattended = unattended.is_unattended(item.track_id)
                    is_fallen = fall.is_fallen(item.track_id)
                    is_interacting = altercation.is_in_interaction(item.track_id)

                    if is_intruding or is_interacting:
                        color = (0, 0, 255)      # red
                    elif is_loitering:
                        color = (0, 165, 255)    # orange
                    elif is_unattended:
                        color = (255, 0, 255)    # magenta
                    elif is_fallen:
                        color = (0, 255, 255)    # yellow
                    else:
                        color = (0, 255, 0)      # green

                    x1, y1, x2, y2 = map(int, item.bbox_xyxy)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                    label = f"{item.class_name} #{item.track_id}"
                    if is_intruding: label += " INTRUSION"
                    if is_loitering: label += " LOITERING"
                    if is_unattended: label += " UNATTENDED"
                    if is_fallen: label += " FALL"
                    if is_interacting: label += " INTERACTION"

                    cv2.putText(vis, label, (x1, max(15, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                draw_weapons(vis, weapon_det.latest)

                cv2.imshow("Smart CCTV - Full Pipeline", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            tracker.prune_stale(frame.timestamp_sec)

    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

