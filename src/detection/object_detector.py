from typing import List
import numpy as np

from ultralytics import YOLO


class Detection:
    def __init__(self, label: str, confidence: float, bbox):
        self.label = label
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)


class ObjectDetector:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        allowed_classes: List[str] | None = None,
    ):
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.allowed_classes = allowed_classes

        # Map class id → name
        self.class_names = self.model.names

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLO inference on a single frame.
        """
        results = self.model(frame, verbose=False)

        detections: List[Detection] = []

        for r in results:
            for box in r.boxes:
                conf = float(box.conf)
                cls_id = int(box.cls)
                label = self.class_names[cls_id]

                if conf < self.confidence_threshold:
                    continue

                if self.allowed_classes and label not in self.allowed_classes:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append(
                    Detection(label, conf, (x1, y1, x2, y2))
                )

        return detections
