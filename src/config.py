"""
Central configuration for the Smart CCTV computer-vision pipeline
(Person Detection -> Multi-Object Tracking -> Restricted-Area Intrusion
 & Loitering Detection), per FYP proposal Section 4.

Keep all tunable parameters here so detection/tracking/activity modules
stay clean and every threshold used in the thesis report / evaluation
chapter is traceable to one place.
"""

from dataclasses import dataclass, field
from pathlib import Path


from typing import Dict, List, Optional


@dataclass
class DetectionConfig:
    # Pretrained YOLOv8 model, configurable detection target classes
    model_path: str = "yolov8n.pt"   # yolov8n.pt or yolov8s.pt
    person_class_id: int = 0
    # COCO classes: 0: person, 24: backpack, 26: handbag, 28: suitcase, 34: baseball bat, 43: knife, 76: scissors
    target_classes: List[int] = field(default_factory=lambda: [0, 24, 26, 28, 34, 43, 76])
    class_names: Dict[int, str] = field(
        default_factory=lambda: {
            0: "person",
            24: "backpack",
            26: "handbag",
            28: "suitcase",
            34: "baseball bat",
            43: "knife",
            76: "scissors",
        }
    )
    confidence_threshold: float = 0.4
    iou_threshold: float = 0.45
    device: str = "cpu"              # "cpu" or "0" for CUDA
    class_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "person": 0.4,
            "backpack": 0.35,
            "handbag": 0.35,
            "suitcase": 0.35,
            "baseball bat": 0.30,
            "knife": 0.28,
            "scissors": 0.28,
        }
    )


@dataclass
class TrackingConfig:
    # Proposal 4.3: ByteTrack (default) or DeepSORT
    tracker: str = "bytetrack"       # "bytetrack" | "deepsort"
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30           # frames to keep a lost track alive
    match_thresh: float = 0.8


@dataclass
class IntrusionConfig:
    # Proposal 4.4: polygon restricted zones, alert on entry
    zones_file: str = "data/zones.json"
    confirm_frames: int = 5


@dataclass
class LoiteringConfig:
    # Proposal 4.5: customizable time threshold
    time_threshold_seconds: float = 30.0
    movement_tolerance_px: float = 40.0


@dataclass
class SuspiciousObjectConfig:
    # Handles both unattended items and dangerous/prohibited object alerts
    enabled: bool = True
    # 1. Unattended / Abandoned items
    unattended_enabled: bool = True
    time_threshold_seconds: float = 15.0
    owner_proximity_px: float = 120.0
    unattended_classes: List[str] = field(
        default_factory=lambda: ["backpack", "handbag", "suitcase"]
    )
    # 2. Dangerous / Prohibited items (weapons, sharp objects)
    dangerous_enabled: bool = True
    dangerous_classes: List[str] = field(
        default_factory=lambda: ["knife", "scissors", "baseball bat"]
    )
    dangerous_confirm_frames: int = 2
    # Combined list for backward compatibility
    classes_to_watch: List[str] = field(
        default_factory=lambda: ["backpack", "handbag", "suitcase", "knife", "scissors", "baseball bat"]
    )


@dataclass
class RTSPConfig:
    # Configuration for IP Camera RTSP streams
    default_url: str = "rtsp://username:password@camera-ip:554/stream1"
    transport: str = "tcp"           # "tcp" prevents UDP packet drop/corruption
    buffer_size: int = 1             # 1 frame buffer prevents live feed lag
    reconnect_delay_sec: float = 2.0
    max_reconnect_attempts: int = 5


@dataclass
class FallDetectionConfig:
    # Observable posture collapse detection (height/width ratio change + vertical velocity)
    enabled: bool = True
    aspect_ratio_threshold: float = 0.85  # W/H > 0.85 indicates horizontal prone posture
    rapid_drop_velocity_px: float = 120.0
    confirm_frames: int = 3


@dataclass
class AltercationConfig:
    # Observable rapid physical motion/interaction indicator
    enabled: bool = True
    proximity_threshold_px: float = 90.0
    velocity_variance_threshold: float = 80.0
    confirm_frames: int = 4

@dataclass
class WeaponConfig:
    # Second YOLO model for weapons (gun, knife, ...). Download with tests/download_weapon_model.py
    enabled: bool = True
    model_path: str = "models/weapon_yolo.pt"
    # Names are matched (case-insensitive) against the model's own class names
    watch_classes: List[str] = field(
        default_factory=lambda: ["gun", "knife", "pistol", "rifle", "short-gun",
                                 "long-gun", "grenade", "explosive", "weapon"]
    )
    min_confidence: float = 0.25          # passed to YOLO (low, filtered per class below)
    confidence_threshold: float = 0.5     # default per-class threshold
    class_thresholds: Dict[str, float] = field(default_factory=dict)  # e.g. {"knife": 0.6}
    imgsz: int = 640
    device: str = "cpu"                   # "0" for CUDA
    run_every_n_frames: int = 3           # CPU-friendly: weapon model runs on every 3rd frame
    draw_hold_frames: int = 6             # keep last boxes on screen this many frames
    confirm_hits: int = 3                 # must be seen in 3 ...
    confirm_window: int = 5               # ... of the last 5 inference runs
    rearm_runs: int = 5                   # runs without the class before it can fire again

@dataclass
class AlertConfig:
    cooldown_seconds: float = 10.0
    evidence_dir: str = "data/evidence"
    sound_alert: bool = False
    email_alert: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    alert_recipient: str = ""


@dataclass
class DatabaseConfig:
    db_path: str = "data/cctv_events.db"


@dataclass
class PreprocessConfig:
    # Standardize frames. Denoise disabled by default to maintain real-time ~25 FPS CPU throughput
    target_width: int = 960
    target_height: int = 540
    denoise: bool = False
    denoise_method: str = "fast_gaussian"  # "none", "fast_gaussian", "nlmeans"
    denoise_strength: int = 7


@dataclass
class EvalConfig:
    iou_match_threshold: float = 0.5


@dataclass
class PipelineConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    intrusion: IntrusionConfig = field(default_factory=IntrusionConfig)
    loitering: LoiteringConfig = field(default_factory=LoiteringConfig)
    suspicious_object: SuspiciousObjectConfig = field(default_factory=SuspiciousObjectConfig)
    fall_detection: FallDetectionConfig = field(default_factory=FallDetectionConfig)
    weapon: WeaponConfig = field(default_factory=WeaponConfig)
    altercation: AltercationConfig = field(default_factory=AltercationConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    rtsp: RTSPConfig = field(default_factory=RTSPConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    project_root: Path = Path(__file__).resolve().parent.parent
    events_log_path: str = "data/events.jsonl"


CONFIG = PipelineConfig()

