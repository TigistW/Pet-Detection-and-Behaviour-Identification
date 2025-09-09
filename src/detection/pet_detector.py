"""YOLO wrapper for pet detection with tracking."""

from ultralytics import YOLO
import numpy as np


class PetDetector:
    def __init__(self, model_path: str = 'models/yolov8n.pt', device: str = 'cpu', conf: float = 0.25):
        """
        Args:
            model_path: path to YOLO model
            device: 'cpu' or 'cuda'
            conf: confidence threshold
        """
        self.model = YOLO(model_path)
        self.device = device
        self.model.to(device)
        self.conf = conf

        # Map COCO class IDs to pet labels
        self.pet_class_ids = {
            15: 'cat',
            16: 'dog',
            17: 'horse',
            18: 'sheep',
            19: 'cow',
            17: 'rabbit',
            20: 'hamster'
        }

    def detect(self, frame: np.ndarray):
        """
        Detect pets in a single BGR frame (OpenCV format) with tracking.

        Returns:
            list of detections: [{"bbox": [x1,y1,x2,y2], "conf": float, "class_id": int, "label": str, "track_id": int}]
        """
        img = frame[..., ::-1]  # BGR -> RGB

        # Use YOLO tracking
        results = self.model.track(img, imgsz=640, conf=self.conf)[0]

        detections = []

        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls = int(box.cls.cpu().numpy())
            conf = float(box.conf.cpu().numpy())
            xyxy = box.xyxy.cpu().numpy().flatten().tolist()
            track_id = int(box.id.cpu().numpy()) if hasattr(box, "id") else None

            # Map COCO class to pet label
            if cls in self.pet_class_ids and conf > 0.5:
                label = self.pet_class_ids[cls]
            else:
                label = "Not Pet"
                cls = -1

            detections.append({
                "bbox": [int(v) for v in xyxy],
                "conf": conf,
                "class_id": cls,
                "label": label,
                "track_id": track_id
            })

        return detections
