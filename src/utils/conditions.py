"""Synthetic condition transforms (brightness, noise, fog...) used for robustness testing.
Used by the live API (condition dropdown) and tests/run_condition_tests.py."""
import cv2
import numpy as np

CONDITIONS = ["none", "dark", "very_dark", "bright", "noise", "blur", "fog", "night", "low_quality"]


def apply_condition(img: np.ndarray, name: str) -> np.ndarray:
    if not name or name == "none":
        return img
    f = img.astype(np.float32)
    if name == "dark":
        out = f * 0.5
    elif name == "very_dark":
        out = f * 0.25
    elif name == "bright":
        out = f * 1.5 + 40
    elif name == "noise":
        out = f + np.random.normal(0, 25, f.shape)
    elif name == "blur":
        return cv2.GaussianBlur(img, (0, 0), 4)
    elif name == "fog":                       # low contrast + haze
        out = f * 0.55 + 255 * 0.45
    elif name == "night":                     # dark, grayscale-ish, noisy (IR camera look)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) * 0.35
        out = np.dstack([g, g, g]) + np.random.normal(0, 12, f.shape)
    elif name == "low_quality":               # heavy JPEG compression + downscale
        small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 12])
        return cv2.resize(cv2.imdecode(buf, 1), (img.shape[1], img.shape[0]))
    else:
        raise ValueError(f"unknown condition: {name}")
    return np.clip(out, 0, 255).astype(np.uint8)
