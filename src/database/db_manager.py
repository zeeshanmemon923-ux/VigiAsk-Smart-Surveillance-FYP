"""
Database Manager for Smart CCTV System

Stores structured CCTV events and alerts in SQLite. Provides parameterized,
injection-safe querying methods for dashboard inspection and natural-language search.
"""

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional


class DatabaseManager:
    def __init__(self, db_path: str = "data/cctv_events.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    datetime_str TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    camera_source TEXT NOT NULL,
                    track_id INTEGER,
                    confidence REAL,
                    zone_name TEXT,
                    duration_seconds REAL,
                    position_x REAL,
                    position_y REAL,
                    evidence_frame_path TEXT,
                    details TEXT,
                    status TEXT DEFAULT 'NEW',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_source);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_track ON events(track_id);
            """)
            conn.commit()

    def insert_event(self, event_data: Dict[str, Any]) -> int:
        """Insert a single event record into the database."""
        ts = float(event_data.get("timestamp", 0.0))
        dt_str = event_data.get("datetime_str")
        if not dt_str:
            dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        pos = event_data.get("position", (0.0, 0.0))
        pos_x = pos[0] if len(pos) > 0 else None
        pos_y = pos[1] if len(pos) > 1 else None

        details_str = json.dumps(event_data.get("details", {}))

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO events (
                    timestamp, datetime_str, event_type, camera_source,
                    track_id, confidence, zone_name, duration_seconds,
                    position_x, position_y, evidence_frame_path, details, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ts,
                dt_str,
                event_data.get("event_type", "unknown"),
                str(event_data.get("camera_source", "0")),
                event_data.get("track_id"),
                float(event_data.get("confidence", 1.0)),
                event_data.get("zone") or event_data.get("zone_name"),
                event_data.get("duration_seconds"),
                pos_x,
                pos_y,
                event_data.get("evidence_frame_path"),
                details_str,
                event_data.get("status", "NEW"),
            ))
            conn.commit()
            return cursor.lastrowid

    def query_events(
        self,
        event_type: Optional[str] = None,
        camera_source: Optional[str] = None,
        track_id: Optional[int] = None,
        min_confidence: Optional[float] = None,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
        min_duration: Optional[float] = None,
        zone_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Safe parameterized query builder preventing SQL injection."""
        query = "SELECT * FROM events WHERE 1=1"
        params: List[Any] = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if camera_source:
            query += " AND camera_source = ?"
            params.append(str(camera_source))

        if track_id is not None:
            query += " AND track_id = ?"
            params.append(track_id)

        if min_confidence is not None:
            query += " AND confidence >= ?"
            params.append(min_confidence)

        if start_datetime:
            query += " AND datetime_str >= ?"
            params.append(start_datetime)

        if end_datetime:
            query += " AND datetime_str <= ?"
            params.append(end_datetime)

        if min_duration is not None:
            query += " AND duration_seconds >= ?"
            params.append(min_duration)

        if zone_name:
            query += " AND zone_name = ?"
            params.append(zone_name)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.query_events(limit=limit)

    def get_event_by_id(self, event_id: int) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events WHERE id = ?", (event_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_event_stats(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM events")
            total = cursor.fetchone()[0]

            cursor.execute("SELECT event_type, COUNT(*) FROM events GROUP BY event_type")
            by_type = dict(cursor.fetchall())

            cursor.execute("SELECT camera_source, COUNT(*) FROM events GROUP BY camera_source")
            by_camera = dict(cursor.fetchall())

            return {
                "total_events": total,
                "by_type": by_type,
                "by_camera": by_camera,
            }
