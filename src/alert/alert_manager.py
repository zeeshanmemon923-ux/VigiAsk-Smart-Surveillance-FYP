"""
Alert Manager for Smart CCTV System

Handles event filtering, deduplication/cooldown, snapshot evidence saving,
audio/email alerts, and persistence to the SQLite database and JSONL log.
"""

from collections import deque
from datetime import datetime
import json
from pathlib import Path
import smtplib
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import AlertConfig
from src.database.db_manager import DatabaseManager


class AlertManager:
    def __init__(
        self,
        alert_cfg: AlertConfig,
        db_manager: Optional[DatabaseManager] = None,
        jsonl_path: Optional[str] = "data/events.jsonl",
    ):
        self.cfg = alert_cfg
        self.db = db_manager or DatabaseManager()
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        self.evidence_dir = Path(self.cfg.evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        # Cooldown state: (event_type, track_id, detail_key) -> last_alert_time
        self._cooldown_state: Dict[Tuple[str, Any, str], float] = {}
        # In-memory alert buffer for fast dashboard streaming
        self.recent_alerts: Deque[Dict[str, Any]] = deque(maxlen=100)

    def process_event(
        self,
        event_dict: Dict[str, Any],
        camera_source: str = "0",
        frame: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Processes a raw event dictionary.
        Returns an enriched alert record if the event passes deduplication, or None.
        """
        event_type = event_dict.get("event_type", "unknown")
        track_id = event_dict.get("track_id")
        if isinstance(track_id, list):
            track_key = "_".join(map(str, track_id))
        else:
            track_key = str(track_id)

        detail_key = event_dict.get("zone", event_dict.get("object_type", ""))
        state_key = (event_type, track_key, detail_key)

        timestamp = float(event_dict.get("timestamp", 0.0))
        last_alert_time = self._cooldown_state.get(state_key, -999.0)

        # Check deduplication window
        if (timestamp - last_alert_time) < self.cfg.cooldown_seconds:
            return None

        # Update cooldown timestamp
        self._cooldown_state[state_key] = timestamp

        now_dt = datetime.now()
        dt_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

        evidence_path_str = None
        if frame is not None:
            filename = f"ev_{event_type}_{track_key}_{now_dt.strftime('%Y%m%d_%H%M%S_%f')[:19]}.jpg"
            evidence_file = self.evidence_dir / filename
            try:
                cv2.imwrite(str(evidence_file), frame)
                evidence_path_str = str(evidence_file)
            except Exception as e:
                print(f"[AlertManager] Error saving evidence frame: {e}")

        alert_record = {
            "timestamp": timestamp,
            "datetime_str": dt_str,
            "event_type": event_type,
            "camera_source": str(camera_source),
            "track_id": event_dict.get("track_id") if isinstance(event_dict.get("track_id"), int) else None,
            "confidence": float(event_dict.get("confidence", 1.0)),
            "zone_name": event_dict.get("zone"),
            "duration_seconds": event_dict.get("duration_seconds"),
            "position": event_dict.get("position", (0.0, 0.0)),
            "evidence_frame_path": evidence_path_str,
            "details": event_dict,
            "status": "NEW",
        }

        # 1. Insert into SQLite Database
        db_id = self.db.insert_event(alert_record)
        alert_record["id"] = db_id

        # 2. Append to JSONL log for backwards compatibility
        if self.jsonl_path:
            try:
                with open(self.jsonl_path, "a") as f:
                    f.write(json.dumps(alert_record) + "\n")
            except Exception as e:
                print(f"[AlertManager] Error logging to JSONL: {e}")

        # 3. Add to in-memory deque
        self.recent_alerts.appendleft(alert_record)

        # 4. Trigger audio alert if configured
        if self.cfg.sound_alert:
            self._trigger_audio()

        # 5. Trigger email notification if configured
        if self.cfg.email_alert and self.cfg.smtp_host and self.cfg.alert_recipient:
            self._dispatch_email(alert_record)

        return alert_record

    def _trigger_audio(self):
        try:
            import winsound
            winsound.Beep(1000, 250)
        except Exception:
            pass

    def _dispatch_email(self, alert_record: Dict[str, Any]):
        try:
            from email.message import EmailMessage
            msg = EmailMessage()
            msg["Subject"] = f"[CCTV Alert] {alert_record['event_type']} on Camera {alert_record['camera_source']}"
            msg["From"] = self.cfg.smtp_user
            msg["To"] = self.cfg.alert_recipient
            body = (
                f"Smart CCTV Alert Triggered\n"
                f"--------------------------\n"
                f"Type: {alert_record['event_type']}\n"
                f"Time: {alert_record['datetime_str']}\n"
                f"Camera: {alert_record['camera_source']}\n"
                f"Track ID: {alert_record['track_id']}\n"
                f"Confidence: {alert_record['confidence']:.2f}\n"
            )
            msg.set_content(body)

            with smtplib.SMTP(self.cfg.smtp_host, self.cfg.smtp_port) as server:
                server.starttls()
                server.login(self.cfg.smtp_user, self.cfg.smtp_password)
                server.send_message(msg)
        except Exception as e:
            print(f"[AlertManager] Failed to dispatch email: {e}")
