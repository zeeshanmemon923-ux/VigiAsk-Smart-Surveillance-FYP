"""Unit tests for weapon event confirmation logic (no model / no GPU needed)."""
from src.activity.weapon_alert import WeaponEventDetector
from src.config import WeaponConfig
from src.detection.weapon_detector import WeaponDetection


class P:  # minimal fake tracked person
    def __init__(self, tid, box):
        self.track_id = tid
        self.bbox_xyxy = box


def gun(conf=0.8, box=(100, 100, 140, 130)):
    return WeaponDetection(*box, conf, "gun")


def test_single_frame_does_not_fire():
    d = WeaponEventDetector(WeaponConfig())
    assert d.update([gun()], [], 0.0) == []


def test_fires_after_confirmation_once():
    cfg = WeaponConfig()
    d = WeaponEventDetector(cfg)
    fired = []
    for i in range(cfg.confirm_hits + 3):
        fired += d.update([gun()], [], i * 0.1)
    assert len(fired) == 1 and fired[0].weapon_class == "gun"


def test_rearms_after_absence():
    cfg = WeaponConfig()
    d = WeaponEventDetector(cfg)
    n = 0
    for i in range(cfg.confirm_hits):
        n += len(d.update([gun()], [], i))
    for i in range(cfg.rearm_runs):
        d.update([], [], 10 + i)
    for i in range(cfg.confirm_hits):
        n += len(d.update([gun()], [], 20 + i))
    assert n == 2


def test_holder_assignment():
    cfg = WeaponConfig()
    d = WeaponEventDetector(cfg)
    ev = []
    for i in range(cfg.confirm_hits):
        ev += d.update([gun()], [P(7, (80, 50, 200, 400)), P(9, (500, 50, 600, 400))], i)
    assert ev[0].holder_track_id == 7


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)