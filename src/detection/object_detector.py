import cv2
from ultralytics import YOLO
from typing import List, Tuple, Dict, Set

class Detection:
    def __init__(self, label: str, confidence: float, bbox):
        self.label = label
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)


class ObjectDetector:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.4,
        roi_padding: int = 10,
        enabled: bool = True
    ):
        self.enabled = enabled
        self.conf_threshold = conf_threshold
        self.roi_padding = roi_padding

        if self.enabled:
            self.model = YOLO(model_name)

    def detect_on_rois(
        self,
        frame,
        rois: List[Tuple[int, int, int, int]]
    ) -> Set[str]:
        """
        Runs YOLO only on motion ROIs.
        Returns a set of detected class names.
        """

        if not self.enabled or not rois:
            return set()

        detections = []
        h, w, _ = frame.shape

        for (x, y, bw, bh) in rois:
            # Add padding
            x1 = max(0, x - self.roi_padding)
            y1 = max(0, y - self.roi_padding)
            x2 = min(w, x + bw + self.roi_padding)
            y2 = min(h, y + bh + self.roi_padding)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            results = self.model(roi, conf=self.conf_threshold, verbose=False)

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id]
                    conf = float(box.conf[0])

                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])

                    detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": (
                        x1 + bx1,
                        y1 + by1,
                        x1 + bx2,
                        y1 + by2
                    )
                })

        return detections
