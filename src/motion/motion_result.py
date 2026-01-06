from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class MotionResult:
    detected: bool
    boxes: List[Tuple[int, int, int, int]]
    total_area: float
