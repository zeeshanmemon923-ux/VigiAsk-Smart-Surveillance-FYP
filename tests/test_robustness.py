"""
System Robustness and Error Handling Test Suite

Validates system resilience when encountering:
1. Missing or non-existent video files
2. Unavailable camera device indices
3. Missing or corrupt zones definition file
4. Missing model weights path
5. Invalid/malformed queries
"""

from pathlib import Path
import sys
import unittest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.activity.intrusion import load_zones
from src.config import CONFIG, DetectionConfig, PreprocessConfig
from src.detection.person_detector import ObjectDetector
from src.search.nl_search import NaturalLanguageSearchEngine
from src.utils.preprocessing import VideoSource


class TestSystemRobustness(unittest.TestCase):

    def test_01_missing_video_file_handling(self):
        """VideoSource must raise an informative IOError when file is missing."""
        cfg = PreprocessConfig()
        with self.assertRaises(IOError) as ctx:
            with VideoSource("data/non_existent_footage_12345.mp4", cfg) as src:
                pass
        self.assertIn("Could not open video source", str(ctx.exception))

    def test_02_unavailable_camera_index(self):
        """VideoSource must raise IOError when invalid camera index is accessed."""
        cfg = PreprocessConfig()
        with self.assertRaises(IOError) as ctx:
            with VideoSource(99, cfg) as src:
                pass
        self.assertIn("Could not open video source", str(ctx.exception))

    def test_03_missing_zones_file_fallback(self):
        """load_zones must safely return an empty list when file is missing."""
        zones = load_zones("data/non_existent_zones_xyz.json")
        self.assertEqual(zones, [])

    def test_04_missing_model_weights_handling(self):
        """ObjectDetector must raise an error when model file does not exist."""
        cfg = DetectionConfig(model_path="models/completely_missing_model.pt")
        with self.assertRaises(Exception):
            _ = ObjectDetector(cfg)

    def test_05_malformed_search_query_handling(self):
        """NaturalLanguageSearchEngine must handle arbitrary strings without crashing."""
        engine = NaturalLanguageSearchEngine()
        weird_queries = [
            "",
            "??? !!! @@@ $$$ %%% ^^^",
            "1234567890",
            "Select from * where 1=1 AND true;",
            "   ",
        ]
        for q in weird_queries:
            results, f = engine.search(q)
            self.assertIsInstance(results, list)
            self.assertIsNotNone(f)


if __name__ == "__main__":
    unittest.main()
