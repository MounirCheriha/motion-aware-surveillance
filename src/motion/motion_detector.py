import cv2
import numpy as np
from typing import List, Tuple
from .motion_result import MotionResult
from .roi_utils import consolidate_boxes

class MotionDetector:
    def __init__(
        self,
        min_area: int = 500,
        history: int = 500,
        detect_shadows: bool = True,
    ):
        self.min_area = min_area

        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=history,
            detectShadows=detect_shadows
        )

        # Morphological kernel for noise removal
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def update(self, frame: np.ndarray) -> MotionResult:
        """
        Process a single frame and detect motion.

        Args:
            frame (np.ndarray): BGR frame

        Returns:
            MotionResult
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        fg_mask = self.back_sub.apply(gray)

        # Remove shadows (MOG2 marks shadows as gray=127)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Clean noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes: List[Tuple[int, int, int, int]] = []
        total_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))
            total_area += area

        detected = len(boxes) > 0
        boxes = consolidate_boxes(boxes, expand=25)

        return MotionResult(
            motion_detected=detected,
            contours=contours,
            bounding_boxes=boxes
        )
