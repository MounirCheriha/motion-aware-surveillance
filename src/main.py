import cv2
import os

from video.reader import VideoReader
from video.writer import EventVideoWriter
from motion.motion_detector import MotionDetector
from events.event_manager import EventManager
from events.metadata_writer import MetadataWriter
from detection.object_detector import ObjectDetector


ENABLE_LABELING = True 
YOLO_FRAME_STRIDE = 5  # run YOLO every N frames


def main():
    reader = VideoReader("data/one-by-one-person-detection.mp4")

    motion_detector = MotionDetector(min_area=800)
    event_manager = EventManager(inactivity_timeout=2.0)

    object_detector = None
    if ENABLE_LABELING:
        object_detector = ObjectDetector(
            model_name="yolov8n.pt",
            conf_threshold=0.5,
            roi_padding=10,
            enabled=ENABLE_LABELING
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
        signal = event_manager.update(motion_result.motion_detected, ts)

        if signal == "start":
            writer.start(event_manager.event_id, frame.shape)
            frame_count = 0

        if event_manager.event_active:

            # Run YOLO only during active events
            if ENABLE_LABELING and (frame_count % YOLO_FRAME_STRIDE == 0):
                detections = object_detector.detect_on_rois(
                    frame,
                    motion_result.bounding_boxes
                )
                for d in detections:
                    x1, y1, x2, y2 = d["bbox"]
                    label = d["label"]
                    conf = d["confidence"]

                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )
                event_manager.add_detections([d["label"] for d in detections])

            # Record frame
            writer.write(frame)
            frame_count += 1

        if signal == "end":
            clip_path = writer.stop()
            
            event_data = event_manager.get_event_metadata(ts)
            event_data["clip_path"] = clip_path

            metadata_writer.add_event(event_data)
            event_manager.reset()
            print(f"Saved metadata for event {event_data['event_id']}")

    metadata_writer.save()
    reader.release()

if __name__ == "__main__":
    main()
