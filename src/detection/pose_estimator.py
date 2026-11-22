"""YOLOv8-Pose wrapper: returns [(bbox_xyxy, keypoints(17,3))] for every person in the frame.
The model (yolov8n-pose.pt) is downloaded automatically on first run."""


class PoseEstimator:
    def __init__(self, model_path="yolov8n-pose.pt", device="cpu", imgsz=640, conf=0.35):
        from ultralytics import YOLO
        self.model, self.device, self.imgsz, self.conf = YOLO(model_path), device, imgsz, conf

    def update(self, frame):
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return []
        boxes, kp = r.boxes.xyxy.cpu().numpy(), r.keypoints.data.cpu().numpy()
        return [(boxes[i], kp[i]) for i in range(len(boxes))]
