"""
RTSP IP Camera Diagnostic & Verification Utility

Tests connectivity to an IP camera over RTSP, validates credentials/stream path,
measures live resolution and FPS, tests frame grabbing, and saves a test snapshot.

Usage:
    py -3.11 tests/test_rtsp_connection.py --url "rtsp://admin:admin123@192.168.1.100:554/stream1"
"""

import argparse
import os
from pathlib import Path
import sys
import time

import cv2

# Set OpenCV FFMPEG options for low-latency TCP RTSP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1024000|max_delay;500000"


def test_rtsp_stream(url: str, snapshot_out: str = "data/rtsp_test_snapshot.jpg") -> bool:
    print(f"\n[RTSP Diagnostic] Testing connection to:")
    # Mask password for secure logging
    safe_url = url
    if "@" in url:
        prefix, host_part = url.split("@", 1)
        if ":" in prefix:
            proto, creds = prefix.split("://", 1)
            user = creds.split(":")[0] if ":" in creds else creds
            safe_url = f"{proto}://{user}:****@{host_part}"
    print(f"  Target: {safe_url}")

    t0 = time.time()
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("  Status: FAILED")
        print("  Error: Could not open RTSP stream.")
        print("\nTroubleshooting Checklist:")
        print("  1. Verify the camera IP is reachable with 'ping <camera-ip>'")
        print("  2. Verify RTSP port 554 is open")
        print("  3. Double check username and password")
        print("  4. Check the stream path (common: /stream1, /ch0_0.h264, /h264Preview_01_main, /onvif1)")
        return False

    # Read a few sample frames to confirm stream stability
    frames_read = 0
    test_frame = None
    for _ in range(10):
        ok, frame = cap.read()
        if ok and frame is not None:
            frames_read += 1
            test_frame = frame
        time.sleep(0.05)

    conn_time = (time.time() - t0) * 1000

    if frames_read == 0 or test_frame is None:
        print("  Status: FAILED (Stream opened but no valid video frames returned)")
        cap.release()
        return False

    h, w = test_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    Path(snapshot_out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(snapshot_out, test_frame)

    print("  Status: SUCCESS")
    print(f"  Resolution: {w}x{h}")
    print(f"  Reported FPS: {fps:.1f}")
    print(f"  Connection Latency: {conn_time:.1f} ms")
    print(f"  Snapshot saved to: {snapshot_out}")
    print(f"  Connected: True\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test IP Camera RTSP Stream")
    parser.add_argument(
        "--url",
        default="rtsp://admin:admin123@192.168.1.100:554/stream1",
        help="Full RTSP URL (e.g. rtsp://user:pass@ip:554/stream1)",
    )
    parser.add_argument("--out", default="data/rtsp_test_snapshot.jpg", help="Output path for snapshot")
    args = parser.parse_args()

    success = test_rtsp_stream(args.url, args.out)
    sys.exit(0 if success else 1)
