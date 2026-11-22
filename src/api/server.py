"""FastAPI backend for the web frontend. Wraps the existing CV pipeline.

Run from the project root:
    pip install fastapi uvicorn
    uvicorn src.api.server:app --port 8000
Open http://localhost:8000
"""
import asyncio
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # config paths (data/...) are relative to the project root

from src.activity.behavior import PROFILES, BehaviorEngine, match_poses
from src.config import CONFIG
from src.database.db_manager import DatabaseManager
from src.search.nl_search import NaturalLanguageSearchEngine
from src.utils.conditions import CONDITIONS, apply_condition

DATA = ROOT / "data"
(DATA / "evidence").mkdir(parents=True, exist_ok=True)
db = DatabaseManager(CONFIG.database.db_path)
search_engine = NaturalLanguageSearchEngine(db)
app = FastAPI(title="Smart CCTV API")


def mask_cam(s) -> str:
    """Never send RTSP credentials to the browser or store them in the DB."""
    return re.sub(r"//[^@/]+@", "//", str(s))


def public(ev: dict) -> dict:
    ev = dict(ev)
    ev["camera_source"] = mask_cam(ev.get("camera_source"))
    p = ev.pop("evidence_frame_path", None)
    ev["evidence_url"] = "/evidence/" + re.split(r"[\\/]", p)[-1] if p else None
    return ev


# ---------------------------------------------------------------- live pipeline
class Runner:
    def __init__(self):
        self.thread = None
        self.stop_evt = threading.Event()
        self.jpeg = None
        self.state = {}
        self.status = {"running": False, "source": None, "condition": "none", "profile": "general", "fps": 0.0, "error": None}

    def start(self, source, label, condition, profile="general"):
        self.stop()
        self.stop_evt.clear()
        self.state = {}
        self.status = {"running": True, "source": label, "condition": condition, "profile": profile, "fps": 0.0, "error": None}
        self.thread = threading.Thread(target=self._run, args=(source, label, condition, profile), daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread and self.thread.is_alive():
            self.stop_evt.set()
            self.thread.join(timeout=10)
        self.status["running"] = False

    def _run(self, source, label, condition, profile):
        try:
            from src.activity.altercation import AltercationDetector
            from src.activity.fall_detection import FallDetector
            from src.activity.intrusion import IntrusionDetector
            from src.activity.loitering import LoiteringDetector
            from src.activity.suspicious_object import SuspiciousObjectDetector
            from src.activity.weapon_alert import WeaponEventDetector
            from src.alert.alert_manager import AlertManager
            from src.detection.pose_estimator import PoseEstimator
            from src.detection.weapon_detector import WeaponDetector, draw_weapons
            from src.tracking.tracker import PersonTracker
            from src.utils.preprocessing import VideoSource

            tracker = PersonTracker(CONFIG.detection, CONFIG.tracking)
            intrusion = IntrusionDetector(CONFIG.intrusion)
            loiter = LoiteringDetector(CONFIG.loitering)
            unatt = SuspiciousObjectDetector(CONFIG.suspicious_object)
            fall = FallDetector(CONFIG.fall_detection)
            alter = AltercationDetector(CONFIG.altercation)
            weapon = WeaponDetector(CONFIG.weapon)
            weapon_ev = WeaponEventDetector(CONFIG.weapon)
            alerts = AlertManager(CONFIG.alert, db, CONFIG.events_log_path)
            pose = PoseEstimator(device=CONFIG.detection.device)
            engine = BehaviorEngine(profile)
            KIND = {"restricted_area_intrusion": "intrusion", "loitering": "loitering",
                    "weapon_detected": "weapon", "high_energy_interaction": "interaction"}

            def feed(d, ts):
                kind = KIND.get(d.get("event_type"))
                ids = d.get("track_id")
                for tid in (ids if isinstance(ids, list) else [ids]):
                    if kind and isinstance(tid, int):
                        engine.external(tid, kind, ts)

            live = isinstance(source, int) or str(source).startswith("rtsp://")
            stride = 3 if live else 2
            t_prev, n = time.time(), 0

            with VideoSource(source, CONFIG.preprocess) as video:
                for frame in video:
                    if self.stop_evt.is_set():
                        break
                    img = apply_condition(frame.image, condition)
                    ts = frame.timestamp_sec
                    items = tracker.update(img, ts, classes=CONFIG.detection.target_classes)
                    people = [i for i in items if i.class_name == "person"]
                    objects = [i for i in items if i.class_name != "person"]

                    events = (intrusion.update(people, ts) + loiter.update(people, tracker.histories, ts)
                              + unatt.update(objects, people, ts) + fall.update(people, tracker.histories, ts)
                              + alter.update(people, tracker.histories, ts))
                    for ev in events:
                        d = ev.to_dict()
                        alerts.process_event(d, label, img)
                        feed(d, ts)

                    wdets = weapon.update(img)
                    if weapon.ran_this_frame:
                        for ev in weapon_ev.update(wdets, people, ts):
                            d = ev.to_dict()
                            alerts.process_event(d, label, draw_weapons(img.copy(), wdets))
                            feed(d, ts)

                    n += 1
                    if n % stride == 0:                              # pose + behaviour every Nth frame (CPU friendly)
                        engine.update(match_poses(people, pose.update(img)), ts)
                    boxes = {i.track_id: i.bbox_xyxy for i in people}
                    for tid in engine.pop_face_requests():           # head crop of the flagged person
                        if tid in boxes:
                            x1, y1, x2, y2 = map(int, boxes[tid])
                            cx, hh = (x1 + x2) // 2, max(int((y2 - y1) * 0.22), 20)
                            crop = img[max(0, y1 - 5):y1 + hh, max(0, cx - hh):cx + hh]
                            if crop.size:
                                if crop.shape[1] < 160:               # upscale tiny crops so the popup is readable
                                    f = 160 / crop.shape[1]
                                    crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
                                fn = f"face_ID{tid}_{int(time.time())}.jpg"
                                cv2.imwrite(str(DATA / "evidence" / fn), crop)
                                engine.set_face(tid, "/evidence/" + fn, ts)
                    for al in engine.pop_alerts():
                        alerts.process_event(al, label, img)

                    vis = img.copy()
                    for z in intrusion.zones:
                        pts = [tuple(map(int, p)) for p in z.polygon]
                        cv2.polylines(vis, [np.array(pts)], True, (0, 0, 255), 2)
                    inside = {t for t, _ in intrusion.currently_inside()}
                    used = []                                     # label boxes already drawn (avoid overlap)
                    for it in items:
                        tags = [n for n, on in (("INTRUSION", it.track_id in inside),
                                                ("LOITERING", loiter.is_loitering(it.track_id)),
                                                ("UNATTENDED", unatt.is_unattended(it.track_id)),
                                                ("FALL", fall.is_fallen(it.track_id)),
                                                ("INTERACTION", alter.is_in_interaction(it.track_id))) if on]
                        st = engine.status(it.track_id, ts) if it.class_name == "person" else "NORMAL"
                        if st in ("SUSPICIOUS", "ALERT"):
                            tags.insert(0, st)
                        color = (0, 0, 255) if tags else (0, 165, 255) if st == "WATCH" else (0, 255, 0)
                        x1, y1, x2, y2 = map(int, it.bbox_xyxy)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                        txt = " ".join([f"#{it.track_id}" if it.class_name == "person" else f"{it.class_name} #{it.track_id}"] + tags)
                        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                        ly = max(th + 6, y1)
                        for _ in range(6):                        # nudge up while overlapping another label
                            r = (x1, ly - th - 6, x1 + tw + 6, ly)
                            if not any(r[0] < u[2] and r[2] > u[0] and r[1] < u[3] and r[3] > u[1] for u in used):
                                break
                            ly = max(th + 6, ly - th - 8)
                        used.append(r)
                        cv2.rectangle(vis, (r[0], r[1]), (r[2], r[3]), color, -1)
                        cv2.putText(vis, txt, (r[0] + 3, r[3] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
                    draw_weapons(vis, weapon.latest)
                    self.state = engine.snapshot(ts, {i.track_id for i in people})
                    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        self.jpeg = buf.tobytes()
                    tracker.prune_stale(ts)

                    now = time.time()
                    self.status["fps"] = round(0.9 * self.status["fps"] + 0.1 / max(now - t_prev, 1e-3), 1)
                    t_prev = now
        except Exception as e:  # surface the error in the UI instead of dying silently
            self.status["error"] = f"{type(e).__name__}: {e}"
        finally:
            self.status["running"] = False


runner = Runner()


class StartReq(BaseModel):
    source: str              # "webcam" | "file:<name in data/>" | "rtsp://..."
    condition: str = "none"
    profile: str = "general"      # general | store | secure_area


@app.get("/api/sources")
def sources():
    return {"videos": sorted(p.name for p in DATA.glob("*.mp4")), "conditions": CONDITIONS, "profiles": list(PROFILES),
            "env_rtsp": bool(os.environ.get("CCTV_RTSP_URL"))}


@app.post("/api/stream/start")
def start(req: StartReq):
    if req.condition not in CONDITIONS:
        raise HTTPException(400, "unknown condition")
    s = req.source
    if s == "webcam":
        src, label = 0, "webcam"
    elif s.startswith("file:"):
        f = (DATA / Path(s[5:]).name)               # .name blocks path traversal
        if not f.exists():
            raise HTTPException(404, "video not found in data/")
        src, label = str(f), f.name
    elif s.startswith("rtsp://"):
        src, label = s, mask_cam(s)
    else:
        raise HTTPException(400, "bad source")
    if req.profile not in PROFILES:
        raise HTTPException(400, "unknown profile")
    runner.start(src, label, req.condition, req.profile)
    return runner.status


@app.post("/api/stream/stop")
def stop():
    runner.stop()
    return runner.status


@app.get("/api/stream/status")
def status():
    return runner.status


@app.get("/api/stream/state")
def state():
    return runner.state


@app.get("/api/stream/video")
def video():
    def gen():
        last = None
        while True:
            j = runner.jpeg
            if j is not None and j is not last:
                last = j
                yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
            time.sleep(0.03)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")


# ---------------------------------------------------------------- events / search
@app.get("/api/events")
def events(type: str = None, limit: int = 100):
    return [public(e) for e in db.query_events(event_type=type, limit=min(limit, 500))]


@app.get("/api/stats")
def stats():
    s = db.get_event_stats()
    s["by_camera"] = {mask_cam(k): v for k, v in s["by_camera"].items()}
    return s


@app.get("/api/search")
def search(q: str):
    rows, f = search_engine.search(q[:300])
    return {"filters": f.to_dict(), "results": [public(r) for r in rows]}


@app.get("/api/alerts/stream")
async def alerts_stream():
    async def gen():
        rows = db.query_events(limit=1)
        last = rows[0]["id"] if rows else 0
        while True:
            for r in sorted(db.query_events(limit=20), key=lambda r: r["id"]):
                if r["id"] > last:
                    last = r["id"]
                    yield f"data: {json.dumps(public(r))}\n\n"
            yield ": ping\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(gen(), media_type="text/event-stream")


app.mount("/evidence", StaticFiles(directory=DATA / "evidence"), name="evidence")
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
