"""
Natural-Language Search Module for Smart CCTV System

Parses ordinary English queries (e.g. "Show restricted-area intrusions yesterday",
"Find events where a person was loitering for more than 2 minutes") into
safe, typed, structured database filters. Guarantees that unrestricted or raw
SQL queries are never executed.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from typing import Any, Dict, List, Optional, Tuple

from src.database.db_manager import DatabaseManager


@dataclass
class SearchFilter:
    event_type: Optional[str] = None
    camera_source: Optional[str] = None
    track_id: Optional[int] = None
    min_confidence: Optional[float] = None
    start_datetime: Optional[str] = None
    end_datetime: Optional[str] = None
    min_duration: Optional[float] = None
    zone_name: Optional[str] = None
    limit: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class NaturalLanguageSearchEngine:
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()

    def parse_query(self, query: str) -> SearchFilter:
        """
        Extracts semantic filters from an English query using pattern matching
        and semantic normalization.
        """
        q = query.lower().strip()
        f = SearchFilter()

        # 1. Event Type Extraction
        if any(w in q for w in ["suspicious", "shoplift", "steal", "stole", "theft", "conceal"]):
            f.event_type = "suspicious_behavior"
        elif any(w in q for w in ["weapon", "gun", "pistol", "rifle", "knife", "firearm", "armed", "grenade"]):
            f.event_type = "weapon_detected"
        elif any(w in q for w in ["intrusion", "restricted", "trespass", "boundary", "enter", "entered", "zone"]):
            f.event_type = "restricted_area_intrusion"
        elif any(w in q for w in ["loiter", "loitering", "linger", "waiting"]):
            f.event_type = "loitering"
        elif any(w in q for w in ["unattended", "abandoned", "bag", "luggage", "backpack", "suitcase", "package"]):
            f.event_type = "unattended_object"
        elif any(w in q for w in ["fall", "falling", "fallen", "collapse", "collapsed", "slip", "trip"]):
            f.event_type = "possible_fall"
        elif any(w in q for w in ["altercation", "fight", "fighting", "interaction", "physical", "motion anomaly"]):
            f.event_type = "high_energy_interaction"

        # 2. Camera Source Extraction
        cam_match = re.search(r"\b(?:camera|cam)\s*(\d+|[a-zA-Z0-9_\-\.]+)", q)
        if cam_match:
            f.camera_source = cam_match.group(1)

        # 3. Track ID Extraction
        track_match = re.search(r"\b(?:track|id|person)\s*#?\s*(\d+)\b", q)
        if track_match:
            f.track_id = int(track_match.group(1))

        # 4. Confidence Extraction
        conf_match = re.search(r"(?:confidence|conf)\s*(?:above|over|>|>=)\s*(\d+(?:\.\d+)?)\s*(%?)", q)
        if conf_match:
            val = float(conf_match.group(1))
            if conf_match.group(2) == "%" or val > 1.0:
                val = val / 100.0
            f.min_confidence = min(1.0, max(0.0, val))
        elif "high confidence" in q:
            f.min_confidence = 0.75

        # 5. Duration Extraction (e.g. "more than 2 minutes", "longer than 30s")
        dur_match = re.search(r"(?:more than|longer than|over|>|>=)\s*(\d+(?:\.\d+)?)\s*(minutes?|mins?|seconds?|secs?|s|m)\b", q)
        if dur_match:
            val = float(dur_match.group(1))
            unit = dur_match.group(2)
            if unit.startswith("m"):
                f.min_duration = val * 60.0
            else:
                f.min_duration = val

        # 6. Zone Name Extraction
        zone_match = re.search(r"\b(zone[_\s]*\d+|[a-zA-Z0-9_\-]+_zone)\b", q)
        if zone_match:
            f.zone_name = zone_match.group(1).replace(" ", "_")

        # 7. Date & Time Range Extraction
        now = datetime.now()
        if "yesterday" in q:
            yesterday = now - timedelta(days=1)
            f.start_datetime = yesterday.strftime("%Y-%m-%d 00:00:00")
            f.end_datetime = yesterday.strftime("%Y-%m-%d 23:59:59")
        elif "today" in q:
            f.start_datetime = now.strftime("%Y-%m-%d 00:00:00")
            f.end_datetime = now.strftime("%Y-%m-%d 23:59:59")
        elif any(w in q for w in ["last 24 hours", "past 24 hours", "past day"]):
            f.start_datetime = (now - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        elif any(w in q for w in ["last hour", "past hour"]):
            f.start_datetime = (now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        elif any(w in q for w in ["this week", "past 7 days", "last 7 days"]):
            f.start_datetime = (now - timedelta(days=7)).strftime("%Y-%m-%d 00:00:00")

        return f

    def search(self, query: str) -> Tuple[List[Dict[str, Any]], SearchFilter]:
        """
        Converts an English query into safe parameterized filters and queries the database.
        Returns matching records and the parsed filter explanation.
        """
        filters = self.parse_query(query)
        results = self.db.query_events(
            event_type=filters.event_type,
            camera_source=filters.camera_source,
            track_id=filters.track_id,
            min_confidence=filters.min_confidence,
            start_datetime=filters.start_datetime,
            end_datetime=filters.end_datetime,
            min_duration=filters.min_duration,
            zone_name=filters.zone_name,
            limit=filters.limit,
        )
        return results, filters
