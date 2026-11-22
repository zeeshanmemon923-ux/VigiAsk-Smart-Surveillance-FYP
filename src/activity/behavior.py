"""
Pose-based behaviour risk engine (scene-independent).

Each tracked person gets a risk score (0-8) built from small observable actions
(reach, hand-to-waist after reach, face hidden) plus signals from the other detectors
(loitering, intrusion, weapon, interaction). A pattern (reach -> conceal) or a score of 8
raises an alert. A scene PROFILE changes the weights and the wording, so the same engine
works for stores, offices, secure areas, parking, etc.

Heuristics on 2D pose: thresholds need tuning on real footage. Alerts say "suspected",
they are a prompt for a human operator, not a verdict.
"""
from collections import deque

import numpy as np

MAX_SCORE, SUSPICIOUS_AT, WATCH_AT, DECAY = 8.0, 4.0, 2.0, 0.15   # decay = points / second

PROFILES = {
    "general":     {"label": "suspicious concealment",
                    "w": {"reach": 0.5, "conceal": 3, "face_hidden": 1, "loitering": 2, "intrusion": 4, "weapon": 6, "interaction": 3}},
    "store":       {"label": "shoplifting",
                    "w": {"reach": 1, "conceal": 3, "face_hidden": 1, "loitering": 2, "intrusion": 3, "weapon": 6, "interaction": 3}},
    "secure_area": {"label": "unauthorised access",
                    "w": {"reach": 0, "conceal": 1, "face_hidden": 2, "loitering": 3, "intrusion": 6, "weapon": 6, "interaction": 4}},
}
NAMES = {"reach": "HIGH REACH", "conceal": "CONCEAL", "face_hidden": "FACE HIDDEN", "loitering": "LOITERING",
         "intrusion": "INTRUSION", "weapon": "WEAPON", "interaction": "INTERACTION"}
TEXT = {"reach": "hand above shoulder", "conceal": "hand to waist after reach", "face_hidden": "face not visible",
        "loitering": "lingering in area", "intrusion": "entered restricted zone", "weapon": "weapon near person",
        "interaction": "rapid physical interaction"}
COOL = {"reach": 3, "conceal": 6, "face_hidden": 10, "loitering": 15, "intrusion": 15, "weapon": 15, "interaction": 8}


class Person:
    def __init__(self, tid, ts):
        self.id, self.score, self.last_ts, self.seen = tid, 0.0, ts, ts
        self.reach_ts, self.conceal_since, self.hidden_since, self.pose_ts = -99.0, None, None, -99.0
        self.cool, self.alert_until, self.face, self.face_asked = {}, -1.0, None, False


class BehaviorEngine:
    def __init__(self, profile="general"):
        self.profile = profile if profile in PROFILES else "general"
        self.cfg = PROFILES[self.profile]
        self.people, self.log = {}, deque(maxlen=30)
        self.alerts, self.banner, self.banner_until = 0, None, -1.0
        self._face_q, self._out = [], []
        self.faces = deque(maxlen=12)          # history of suspect snapshots

    # ---- helpers
    def _get(self, tid, ts):
        p = self.people.get(tid)
        if p is None:
            p = self.people[tid] = Person(tid, ts)
        return p

    def _add(self, p, kind, ts, text=None):
        w = self.cfg["w"].get(kind, 0)
        if w <= 0 or ts - p.cool.get(kind, -99) < COOL[kind]:
            return False
        p.cool[kind] = ts
        p.score = min(MAX_SCORE, p.score + w)
        self.log.appendleft({"ts": ts, "id": p.id, "kind": kind, "text": text or TEXT[kind], "name": NAMES[kind]})
        if p.score >= SUSPICIOUS_AT and not p.face_asked:
            p.face_asked = True
            self._face_q.append(p.id)
        if p.score >= MAX_SCORE:
            self._alert(p, ts, "high-risk behaviour")
        return True

    def _alert(self, p, ts, why):
        if ts - p.cool.get("alert", -99) < 15:
            return
        p.cool["alert"], p.alert_until = ts, ts + 10
        self.alerts += 1
        if not p.face_asked:
            p.face_asked = True
            self._face_q.append(p.id)
        self.banner, self.banner_until = f"ID {p.id} suspected {self.cfg['label']} · {why}", ts + 8
        self._out.append({"event_type": "suspicious_behavior", "track_id": p.id, "timestamp": ts,
                          "confidence": round(min(1.0, p.score / MAX_SCORE), 2),
                          "profile": self.profile, "label": self.cfg["label"], "reason": why})

    # ---- inputs
    def update(self, poses, ts):
        """poses: [(track_id, bbox_xyxy, keypoints(17,3) COCO order)]"""
        for tid, box, k in poses:
            p = self._get(tid, ts)
            p.score = max(0.0, p.score - DECAY * (ts - p.last_ts))
            p.last_ts = p.seen = p.pose_ts = ts
            h = max(float(box[3] - box[1]), 1.0)
            ok = lambda i, c=0.4: k[i][2] >= c
            sh = [k[i][1] for i in (5, 6) if ok(i)]
            if not sh:
                continue
            sh_y = float(np.mean(sh))

            if any(ok(i) and k[i][1] < sh_y - 0.05 * h for i in (9, 10)):      # wrist above shoulder
                p.reach_ts = ts
                self._add(p, "reach", ts)

            in_waist = False
            if ok(11) and ok(12) and ts - p.reach_ts <= 6:                      # only counts right after a reach
                mid_x, hip_y = (k[11][0] + k[12][0]) / 2, (k[11][1] + k[12][1]) / 2
                half = abs(k[11][0] - k[12][0]) / 2
                in_waist = any(ok(i) and abs(k[i][1] - hip_y) < 0.15 * h
                               and abs(k[i][0] - mid_x) < 0.8 * max(half, 0.05 * h) for i in (9, 10))
            if in_waist:                                                         # hand must STAY inside the hip line
                if p.conceal_since is None:
                    p.conceal_since = ts
                if ts - p.conceal_since >= 0.5 and self._add(p, "conceal", ts):
                    self._alert(p, ts, "reach → hand to waist")
            else:
                p.conceal_since = None

            if not any(ok(i, 0.3) for i in (0, 1, 2)):                          # no nose/eyes visible
                if p.hidden_since is None:
                    p.hidden_since = ts
                if ts - p.hidden_since >= 1.5:
                    self._add(p, "face_hidden", ts)
            else:
                p.hidden_since = None

    def external(self, tid, kind, ts, text=None):
        """Feed events from other detectors (loitering, intrusion, weapon, interaction)."""
        if tid is None:
            return
        p = self._get(tid, ts)
        p.seen = ts
        self._add(p, kind, ts, text)

    # ---- outputs
    def pop_alerts(self):
        out, self._out = self._out, []
        return out

    def pop_face_requests(self):
        out, self._face_q = self._face_q, []
        return out

    def set_face(self, tid, url, ts=0.0):
        if tid in self.people:
            self.people[tid].face = url
            self.faces.appendleft({"id": tid, "url": url, "ts": ts})

    def status(self, tid, ts):
        p = self.people.get(tid)
        if p is None:
            return "NORMAL"
        if ts < p.alert_until:
            return "ALERT"
        return "SUSPICIOUS" if p.score >= SUSPICIOUS_AT else "WATCH" if p.score >= WATCH_AT else "NORMAL"

    def snapshot(self, ts, tracked=None):
        """tracked = ids of ALL people the tracker currently sees (pose or not)."""
        ids = set(tracked or ()) | {p.id for p in self.people.values() if ts - p.seen <= 3}
        rows = []
        for tid in ids:
            p = self.people.get(tid)
            has_pose = p is not None and ts - p.pose_ts <= 3
            status = self.status(tid, ts)
            if not has_pose and status == "NORMAL":
                status = "NO_POSE"                     # tracked, but too small/far for pose estimation
            rows.append({"id": tid, "status": status, "score": round(p.score if p else 0.0, 1),
                         "max": MAX_SCORE, "pose": has_pose})
        rows.sort(key=lambda r: (-r["score"], not r["pose"], r["id"]))
        sus = next((self.people[r["id"]] for r in rows
                    if r["id"] in self.people and self.people[r["id"]].face and r["status"] in ("SUSPICIOUS", "ALERT")), None)
        return {
            "profile": self.profile,
            "banner": self.banner if ts < self.banner_until else None,
            "stats": {"people": len(tracked) if tracked is not None else len(rows), "alerts": self.alerts,
                      "suspicious": sum(r["status"] in ("SUSPICIOUS", "ALERT") for r in rows)},
            "people": rows,
            "log": [{"id": e["id"], "name": e["name"], "text": e["text"], "kind": e["kind"], "ago": round(ts - e["ts"], 1)}
                    for e in list(self.log)[:8]],
            "suspect": {"id": sus.id, "url": sus.face} if sus else None,
            "faces": [{"id": f["id"], "url": f["url"], "ago": round(ts - f["ts"], 1)} for f in list(self.faces)[:6]],
        }


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match_poses(items, poses, thr=0.3):
    """Attach pose detections [(bbox, kpts)] to tracker items (track_id, bbox_xyxy) by IoU."""
    pairs = sorted(((_iou(it.bbox_xyxy, b), i, j) for i, it in enumerate(items) for j, (b, _) in enumerate(poses)),
                   reverse=True)
    used_i, used_j, out = set(), set(), []
    for s, i, j in pairs:
        if s < thr:
            break
        if i not in used_i and j not in used_j:
            used_i.add(i); used_j.add(j)
            out.append((items[i].track_id, poses[j][0], poses[j][1]))
    return out
