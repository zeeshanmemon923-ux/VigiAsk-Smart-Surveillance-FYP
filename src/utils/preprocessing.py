"""
Data Preprocessing (Proposal Section 4.1)

Handles reading recorded footage or a live stream, and standardizing
frames before they hit the detector: resize, denoise, color-space
conversion. Keeping this isolated means the detector never has to
care whether the source was a file, RTSP stream, or webcam.
"""

from dataclasses import dataclass
import time
from typing import Iterator, Optional

import os
import threading

# Low-latency RTSP: TCP transport, no input buffering. Set before any VideoCapture is created.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

import cv2
import numpy as np

from src.config import PreprocessConfig


@dataclass
class Frame:
    """A single preprocessed frame plus its metadata."""
    index: int
    timestamp_sec: float
    image: np.ndarray          # BGR, resized, denoised — ready for the detector
    raw_image: np.ndarray      # original resolution, kept for display/recording


class VideoSource:
    """
    Wraps cv2.VideoCapture for either a file path, an RTSP/HTTP stream
    URL, or a webcam index, and yields preprocessed Frame objects.

    Usage:
        with VideoSource("data/sample.mp4", cfg) as src:
            for frame in src:
                ...
    """

    def __init__(self, source, cfg: PreprocessConfig):
        # source can be: str path, str URL, or int webcam index
        self.source = source
        self.cfg = cfg
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_index = 0
        # Live = RTSP stream or webcam. Files are not live.
        self._is_rtsp = isinstance(source, str) and source.lower().startswith("rtsp://")
        self._live = self._is_rtsp or isinstance(source, int)
        self._t0 = None
        self.max_reconnects = 10          # RTSP only: give up after this many failed retries
        # Live sources are read by a background thread that always keeps only the NEWEST frame,
        # so slow inference drops frames instead of building up seconds of delay.
        self._latest, self._lock = None, threading.Lock()
        self._new, self._stop, self._ended, self._thread = threading.Event(), False, False, None

    def _open(self) -> cv2.VideoCapture:
        if self._is_rtsp:
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep latency low
        else:
            cap = cv2.VideoCapture(self.source)
        return cap

    def __enter__(self) -> "VideoSource":
        self._cap = self._open()
        if not self._cap.isOpened():
            raise IOError(f"Could not open video source: {self.source}")
        self._t0 = time.monotonic()
        if self._live:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._cap is not None:
            self._cap.release()

    def _reader(self):
        """Background thread: read as fast as the camera sends, keep only the latest frame."""
        while not self._stop:
            ok, raw = self._cap.read()
            if not ok:
                if not self._is_rtsp:
                    break
                ok = False
                for attempt in range(1, self.max_reconnects + 1):   # stream dropped: reconnect
                    if self._stop:
                        return
                    print(f"[VideoSource] Lost RTSP connection, reconnecting ({attempt}/{self.max_reconnects})...")
                    self._cap.release()
                    time.sleep(2)
                    self._cap = self._open()
                    ok, raw = self._cap.read()
                    if ok:
                        break
                if not ok:
                    break
            with self._lock:
                self._latest = raw
            self._new.set()
        self._ended = True
        self._new.set()

    def _next_live(self):
        waited = 0.0
        while True:
            if not self._new.wait(timeout=1.0):
                waited += 1.0
                if self._ended or waited >= 15:      # stream ended or silent for 15 s
                    raise StopIteration
                continue
            self._new.clear()
            with self._lock:
                raw, self._latest = self._latest, None
            if raw is not None:
                return raw
            if self._ended:
                raise StopIteration

    @property
    def fps(self) -> float:
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 30.0

    def __iter__(self) -> Iterator[Frame]:
        return self

    def __next__(self) -> Frame:
        assert self._cap is not None, "Use VideoSource as a context manager"
        if self._live:
            raw = self._next_live()
        else:
            ok, raw = self._cap.read()
            if not ok:
                raise StopIteration
        processed = preprocess_frame(raw, self.cfg)
        # Live sources use wall-clock time so loitering / unattended timers stay correct
        # even when processing is slower than the camera frame rate.
        ts = (time.monotonic() - self._t0) if self._live else (self._frame_index / self.fps)
        frame = Frame(
            index=self._frame_index,
            timestamp_sec=ts,
            image=processed,
            raw_image=raw,
        )
        self._frame_index += 1
        return frame


def preprocess_frame(frame: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    Standardize one raw BGR frame per proposal 4.1:
    resize -> (optional) denoise. Color space is kept BGR since that's
    what YOLO/OpenCV expect; convert only where a specific module needs
    a different space (e.g. HSV for a future module).
    """
    resized = cv2.resize(
        frame, (cfg.target_width, cfg.target_height), interpolation=cv2.INTER_AREA
    )
    if cfg.denoise:
        method = getattr(cfg, "denoise_method", "fast_gaussian")
        if method == "fast_gaussian":
            resized = cv2.GaussianBlur(resized, (3, 3), 0)
        elif method == "fast_bilateral":
            resized = cv2.bilateralFilter(resized, 5, 40, 40)
        elif method == "nlmeans":
            resized = cv2.fastNlMeansDenoisingColored(
                resized, None, cfg.denoise_strength, cfg.denoise_strength, 7, 21
            )
    return resized


def scale_point_to_raw(point, processed_shape, raw_shape):
    """
    Map an (x, y) point in the resized/processed frame back to
    raw-frame coordinates, e.g. for drawing on the original-resolution
    frame or for zone-intrusion checks defined in raw coordinates.
    """
    px, py = point
    ph, pw = processed_shape[:2]
    rh, rw = raw_shape[:2]
    return px * (rw / pw), py * (rh / ph)
