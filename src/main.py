import cv2
import os

from video.reader import VideoReader
from video.writer import EventVideoWriter
from motion.motion_detector import MotionDetector
from events.event_manager import EventManager
from detection.object_detector import ObjectDetector


def main():
    reader = VideoReader("data/sample_video.mp4")

    motion_detector = MotionDetector(min_area=800)
    event_manager = EventManager(inactivity_timeout=2.0)

    object_detector = ObjectDetector(
        model_name="yolov8n.pt",
        confidence_threshold=0.5,
        allowed_classes=None
    )

    writer = EventVideoWriter(
        output_dir="outputs/events",
        fps=reader.fps
    )

    for frame, ts in reader.read():
        motion_result = motion_detector.update(frame)
        signal = event_manager.update(motion_result.detected, ts)

        if signal == "start":
            writer.start(event_manager.event_id, frame.shape)

        if event_manager.event_active:
            # Record frame
            writer.write(frame)

            # Run YOLO only during active events
            detections = object_detector.detect(frame)
            event_manager.add_detections(detections)
            
        if signal == "end":
            path = writer.stop()
            if path:
                print(f"Saved event video: {path}")

    reader.release()

if __name__ == "__main__":
    main()
