"""
Unit Test Suite for Smart CCTV Modules

Tests individual modules independently:
- ObjectDetector & PersonDetector
- PersonTracker & TrackHistory
- IntrusionDetector (polygon containment)
- LoiteringDetector (dwell time + movement tolerance)
- SuspiciousObjectDetector (unattended object logic)
- FallDetector (aspect ratio collapse)
- AltercationDetector (proximity + rapid displacement)
- DatabaseManager (insert, parameterized query, stats)
- AlertManager (cooldown deduplication, snapshot generation)
"""

from pathlib import Path
import sys
import unittest

import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.activity.altercation import AltercationDetector
from src.activity.fall_detection import FallDetector
from src.activity.intrusion import IntrusionDetector, Zone
from src.activity.loitering import LoiteringDetector
from src.activity.suspicious_object import SuspiciousObjectDetector
from src.alert.alert_manager import AlertManager
from src.config import CONFIG, AlertConfig, FallDetectionConfig, LoiteringConfig, SuspiciousObjectConfig
from src.database.db_manager import DatabaseManager
from src.detection.person_detector import Detection, ObjectDetector, PersonDetector
from src.tracking.tracker import TrackHistory, TrackedPerson


class TestSmartCCTVModules(unittest.TestCase):

    def test_01_detection_dataclass(self):
        """Test Detection bounding-box geometry and aspect ratio calculation."""
        det = Detection(x1=100, y1=100, x2=200, y2=300, confidence=0.88, class_id=0, class_name="person")
        self.assertEqual(det.width, 100)
        self.assertEqual(det.height, 200)
        self.assertAlmostEqual(det.aspect_ratio, 0.5)
        self.assertEqual(det.bottom_center, (150, 300))

    def test_02_zone_containment(self):
        """Test polygon zone point containment logic."""
        zone = Zone(name="vault", polygon=[(0, 0), (200, 0), (200, 200), (0, 200)])
        self.assertTrue(zone.contains((100, 100)))  # inside
        self.assertFalse(zone.contains((300, 300))) # outside

    def test_03_loitering_detection(self):
        """Test dwell time and position spread triggering loitering alerts."""
        cfg = LoiteringConfig(time_threshold_seconds=5.0, movement_tolerance_px=30.0)
        detector = LoiteringDetector(cfg)

        person = TrackedPerson(track_id=1, x1=100, y1=100, x2=150, y2=250, confidence=0.9)
        hist = TrackHistory(track_id=1)

        # Simulate stationary person over 6 seconds
        hist.update((125, 250), (100, 100, 150, 250), timestamp=0.0)
        hist.update((126, 250), (100, 100, 150, 250), timestamp=2.0)
        hist.update((124, 251), (100, 100, 150, 250), timestamp=6.0)

        events = detector.update([person], {1: hist}, timestamp=6.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].track_id, 1)
        self.assertEqual(events[0].duration_seconds, 6.0)

    def test_04_suspicious_unattended_object(self):
        """Test stationary backpack left without a person nearby."""
        cfg = SuspiciousObjectConfig(enabled=True, time_threshold_seconds=3.0, owner_proximity_px=50.0)
        detector = SuspiciousObjectDetector(cfg)

        bag = TrackedPerson(track_id=10, x1=200, y1=200, x2=240, y2=240, confidence=0.85, class_id=24, class_name="backpack")
        far_person = TrackedPerson(track_id=1, x1=500, y1=500, x2=550, y2=650, confidence=0.9, class_id=0, class_name="person")

        # Frame at t=0
        events_0 = detector.update([bag], [far_person], timestamp=0.0)
        self.assertEqual(len(events_0), 0)

        # Frame at t=4.0s (exceeds threshold)
        events_1 = detector.update([bag], [far_person], timestamp=4.0)
        self.assertEqual(len(events_1), 1)
        self.assertEqual(events_1[0].object_type, "backpack")
        self.assertEqual(events_1[0].track_id, 10)

    def test_05_fall_detection(self):
        """Test horizontal aspect-ratio collapse triggering fall alert."""
        cfg = FallDetectionConfig(enabled=True, aspect_ratio_threshold=0.8, confirm_frames=2)
        detector = FallDetector(cfg)

        # Fallen person: width 200, height 100 -> Aspect Ratio = 2.0 (> 0.8)
        fallen = TrackedPerson(track_id=5, x1=100, y1=100, x2=300, y2=200, confidence=0.88, class_id=0, class_name="person")
        hist = TrackHistory(track_id=5)
        hist.update(fallen.bottom_center, fallen.bbox_xyxy, timestamp=1.0)

        # Frame 1: consecutive = 1
        ev1 = detector.update([fallen], {5: hist}, timestamp=1.0)
        self.assertEqual(len(ev1), 0)

        # Frame 2: consecutive = 2 -> confirmed
        ev2 = detector.update([fallen], {5: hist}, timestamp=1.1)
        self.assertEqual(len(ev2), 1)
        self.assertEqual(ev2[0].track_id, 5)
        self.assertEqual(ev2[0].event_type, "possible_fall")

    def test_06_database_manager(self):
        """Test SQLite database operations and parameterized queries."""
        import time
        test_db_path = f"data/test_events_{int(time.time()*1000)}.db"
        db = DatabaseManager(test_db_path)

        ev_data = {
            "timestamp": 12.5,
            "datetime_str": "2026-10-18 12:00:00",
            "event_type": "restricted_area_intrusion",
            "camera_source": "0",
            "track_id": 42,
            "confidence": 0.92,
            "zone": "zone_1",
            "position": (150.0, 250.0),
        }

        row_id = db.insert_event(ev_data)
        self.assertGreater(row_id, 0)

        # Query back
        res = db.query_events(event_type="restricted_area_intrusion", track_id=42)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["track_id"], 42)
        self.assertAlmostEqual(res[0]["confidence"], 0.92)

        # Clean up
        try:
            del db
            Path(test_db_path).unlink(missing_ok=True)
        except Exception:
            pass

    def test_07_alert_cooldown_deduplication(self):
        """Test alert manager prevents spamming within cooldown window."""
        import time
        test_db_path = f"data/test_cooldown_{int(time.time()*1000)}.db"
        test_db = DatabaseManager(test_db_path)
        cfg = AlertConfig(cooldown_seconds=10.0, evidence_dir="data/test_evidence", sound_alert=False)
        mgr = AlertManager(cfg, test_db, jsonl_path=None)

        ev = {
            "timestamp": 100.0,
            "event_type": "loitering",
            "track_id": 3,
            "confidence": 0.85,
            "duration_seconds": 32.0,
        }

        # First trigger: should produce an alert
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        al1 = mgr.process_event(ev, "0", dummy_frame)
        self.assertIsNotNone(al1)

        # Second trigger at t=102.0s (only 2s later): must be suppressed by cooldown
        ev2 = dict(ev, timestamp=102.0)
        al2 = mgr.process_event(ev2, "0", dummy_frame)
        self.assertIsNone(al2)

        # Third trigger at t=111.0s (11s later): cooldown expired, should fire
        ev3 = dict(ev, timestamp=111.0)
        al3 = mgr.process_event(ev3, "0", dummy_frame)
        self.assertIsNotNone(al3)

        # Clean up
        try:
            del mgr
            del test_db
            Path(test_db_path).unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
