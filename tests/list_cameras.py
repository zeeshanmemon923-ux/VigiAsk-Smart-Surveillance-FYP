"""
Lists available camera indices by trying to open each one briefly.
Useful when you have more than one camera (e.g. laptop cam + USB
webcam) and need to find which index OpenCV assigns to which.

Usage:
    python tests/list_cameras.py
"""

import cv2

print("Checking camera indices 0-4...")
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW is faster/more reliable on Windows
    if cap.isOpened():
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            print(f"  index {i}: WORKS ({w}x{h})")
        else:
            print(f"  index {i}: opens but no frame")
        cap.release()
    else:
        print(f"  index {i}: not available")
