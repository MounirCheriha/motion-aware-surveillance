import pytest
import numpy as np


@pytest.fixture
def blank_frame():
    # return a small black BGR frame
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def frame_with_rect(blank_frame):
    f = blank_frame.copy()
    # draw a white rectangle
    import cv2
    cv2.rectangle(f, (10, 10), (40, 30), (255, 255, 255), -1)
    return f
