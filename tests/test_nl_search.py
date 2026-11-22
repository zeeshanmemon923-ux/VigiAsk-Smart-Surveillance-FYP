"""
Evaluation Suite for Natural-Language Search Engine

Tests retrieval accuracy, query semantic translation, parameter binding,
and security resilience against SQL injection attempts.
"""

from pathlib import Path
import sys
import unittest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.database.db_manager import DatabaseManager
from src.search.nl_search import NaturalLanguageSearchEngine, SearchFilter


class TestNaturalLanguageSearch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_db_path = "data/test_search.db"
        Path(cls.test_db_path).unlink(missing_ok=True)
        cls.db = DatabaseManager(cls.test_db_path)
        cls.engine = NaturalLanguageSearchEngine(cls.db)

        # Seed realistic CCTV incident records
        cls.db.insert_event({
            "timestamp": 10.0,
            "datetime_str": "2026-10-18 10:00:00",
            "event_type": "restricted_area_intrusion",
            "camera_source": "0",
            "track_id": 1,
            "confidence": 0.95,
            "zone": "zone_1",
            "duration_seconds": None,
        })
        cls.db.insert_event({
            "timestamp": 50.0,
            "datetime_str": "2026-10-18 10:15:00",
            "event_type": "loitering",
            "camera_source": "0",
            "track_id": 2,
            "confidence": 0.82,
            "zone": None,
            "duration_seconds": 150.0,  # 2.5 minutes
        })
        cls.db.insert_event({
            "timestamp": 80.0,
            "datetime_str": "2026-10-18 10:30:00",
            "event_type": "unattended_object",
            "camera_source": "1",
            "track_id": 10,
            "confidence": 0.88,
            "zone": None,
            "duration_seconds": 45.0,
        })
        cls.db.insert_event({
            "timestamp": 120.0,
            "datetime_str": "2026-10-18 11:00:00",
            "event_type": "possible_fall",
            "camera_source": "0",
            "track_id": 5,
            "confidence": 0.78,
            "zone": None,
            "duration_seconds": None,
        })

    @classmethod
    def tearDownClass(cls):
        try:
            del cls.engine
            del cls.db
            Path(cls.test_db_path).unlink(missing_ok=True)
        except Exception:
            pass

    def test_01_intrusion_query(self):
        """Query: 'Show all restricted-area intrusions'"""
        results, f = self.engine.search("Show all restricted-area intrusions")
        self.assertEqual(f.event_type, "restricted_area_intrusion")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["track_id"], 1)

    def test_02_loitering_duration_query(self):
        """Query: 'Find events where a person was loitering for more than 2 minutes'"""
        results, f = self.engine.search("Find events where a person was loitering for more than 2 minutes")
        self.assertEqual(f.event_type, "loitering")
        self.assertEqual(f.min_duration, 120.0)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["track_id"], 2)

    def test_03_camera_source_filter(self):
        """Query: 'Show unattended luggage on camera 1'"""
        results, f = self.engine.search("Show unattended luggage on camera 1")
        self.assertEqual(f.event_type, "unattended_object")
        self.assertEqual(f.camera_source, "1")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["track_id"], 10)

    def test_04_confidence_filter(self):
        """Query: 'Show events with confidence above 0.9'"""
        results, f = self.engine.search("Show events with confidence above 0.9")
        self.assertAlmostEqual(f.min_confidence, 0.9)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["track_id"], 1)

    def test_05_track_id_filter(self):
        """Query: 'Find incidents for track 5'"""
        results, f = self.engine.search("Find incidents for track 5")
        self.assertEqual(f.track_id, 5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["event_type"], "possible_fall")

    def test_06_sql_injection_defense(self):
        """Verify adversarial SQL injection inputs are treated safely as literals."""
        malicious_query = "'; DROP TABLE events; --"
        results, f = self.engine.search(malicious_query)
        # Verify events table still exists and functions
        stats = self.db.get_event_stats()
        self.assertEqual(stats["total_events"], 4)


if __name__ == "__main__":
    unittest.main()
