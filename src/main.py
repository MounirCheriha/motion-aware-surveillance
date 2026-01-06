import cv2
import os

from video.reader import VideoReader
from video.writer import EventVideoWriter
from motion.motion_detector import MotionDetector
from events.event_manager import EventManager
from events.metadata_writer import MetadataWriter
from detection.object_detector import ObjectDetector


ENABLE_LABELING = True 
YOLO_FRAME_STRIDE = 15  # run YOLO every N frames


def main():
    reader = VideoReader("data/sample_video.mp4")

    motion_detector = MotionDetector(min_area=800)
    event_manager = EventManager(inactivity_timeout=2.0)

    object_detector = None
    if ENABLE_LABELING:
        object_detector = ObjectDetector(
            model_name="yolov8n.pt",
            confidence_threshold=0.5,
            allowed_classes=None
        )

    writer = EventVideoWriter(
        output_dir="outputs/events",
        fps=reader.fps
    )
    metadata_writer = MetadataWriter(
        output_path="outputs/metadata/events.json"
    )

    frame_count = 0
    for frame, ts in reader.read():
        motion_result = motion_detector.update(frame)
        signal = event_manager.update(motion_result.detected, ts)

        if signal == "start":
            writer.start(event_manager.event_id, frame.shape)
            frame_count = 0

        if event_manager.event_active:
            # Record frame
            writer.write(frame)

            # Run YOLO only during active events
            if ENABLE_LABELING and (frame_count % YOLO_FRAME_STRIDE == 0):
                detections = object_detector.detect(frame)
                event_manager.add_detections(detections)

            frame_count += 1

        if signal == "end":
            clip_path = writer.stop()
            
            event_data = event_manager.get_event_metadata(ts)
            event_data["clip_path"] = clip_path

            metadata_writer.add_event(event_data)
            print(f"Saved metadata for event {event_data['event_id']}")

    metadata_writer.save()
    reader.release()

if __name__ == "__main__":
    main()
