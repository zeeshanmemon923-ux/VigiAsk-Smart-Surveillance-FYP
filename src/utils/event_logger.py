"""
Event Logger

Appends structured events (intrusion, loitering, ...) to a JSONL file.
This is the hand-off point to Aqib's backend: he can tail/ingest
data/events.jsonl into Postgres/Mongo, or later we replace this with a
direct POST to his FastAPI endpoint without touching detection code.
"""

import json
from pathlib import Path


class EventLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_dict: dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(event_dict) + "\n")
