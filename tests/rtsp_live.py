import os

# Force TCP for a stable stream (must be set before importing cv2)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"

import time
import cv2

# stream1 = main stream. If it lags, try the sub stream: .../stream2
RTSP_URL = "rtsp://admin:admin123@192.168.100.2:554/stream1"
WINDOW = "Live Stream (press q to quit, s to save snapshot)"


def open_stream():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


cap = open_stream()
print("Connecting...")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Lost connection, reconnecting in 2 seconds...")
        cap.release()
        time.sleep(2)
        cap = open_stream()
        continue

    cv2.imshow(WINDOW, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        name = time.strftime("snapshot_%Y%m%d_%H%M%S.jpg")
        cv2.imwrite(name, frame)
        print("Saved", name)

cap.release()
cv2.destroyAllWindows()
