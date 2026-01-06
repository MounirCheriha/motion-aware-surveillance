import cv2
import os

from video.reader import VideoReader
from video.writer import EventVideoWriter
from motion.motion_detector import MotionDetector
from events.event_manager import EventManager


def main():
    reader = VideoReader("data/sample_video.mp4")
    detector = MotionDetector(min_area=800)
    event_manager = EventManager(inactivity_timeout=2.0)

    writer = EventVideoWriter(
        output_dir="outputs/events",
        fps=reader.fps
    )

    for frame, ts in reader.read():
        motion_result = detector.update(frame)
        signal = event_manager.update(motion_result.detected, ts)

        if signal == "start":
            writer.start(event_manager.event_id, frame.shape)

        if event_manager.event_active:
            writer.write(frame)

        if signal == "end":
            path = writer.stop()
            if path:
                print(f"Saved event video: {path}")

    reader.release()

if __name__ == "__main__":
    main()
