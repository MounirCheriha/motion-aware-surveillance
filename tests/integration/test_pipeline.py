from src.motion.motion_detector import MotionDetector
from src.detection.object_detector import ObjectDetector


def test_pipeline_motion_then_detection(blank_frame, frame_with_rect):
    # Motion detector with short history
    md = MotionDetector(min_area=50, history=2)

    for _ in range(10):
        md.update(blank_frame)

    mr = md.update(frame_with_rect)

    assert hasattr(mr, "motion_detected")

    # Object detector disabled to avoid loading heavy models
    od = ObjectDetector(enabled=False)
    detections = od.detect_on_rois(frame_with_rect, mr.bounding_boxes)

    # With detector disabled, should be empty set/list
    assert detections == set() or detections == []
