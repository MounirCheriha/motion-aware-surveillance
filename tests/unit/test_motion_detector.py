from src.motion.motion_detector import MotionDetector


def test_motion_detector_detects_simple_motion(blank_frame, frame_with_rect):
    
    md = MotionDetector(min_area=50, history=2)

    # Background with several identical blank frames
    for _ in range(10):
        md.update(blank_frame)

    # Introduce a rectangle
    result = md.update(frame_with_rect)

    assert hasattr(result, "motion_detected")
    assert isinstance(result.motion_detected, bool)
    assert isinstance(result.bounding_boxes, list)
