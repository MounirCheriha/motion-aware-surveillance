from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class MotionResult:
    motion_detected: bool
    contours: List
    bounding_boxes: List[Tuple[int, int, int, int]]