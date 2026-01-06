import cv2
import os

from video.reader import VideoReader
from motion.motion_detector import MotionDetector
from events.event_manager import EventManager

DEBUG_VISUALIZE = False

def draw_motion_boxes(frame, boxes):
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def main():
    reader = VideoReader("data/sample_video.mp4")
    detector = MotionDetector(min_area=800)
    event_manager = EventManager(inactivity_timeout=2.0)

    frame_idx = 0

    for frame, ts in reader.read():
        motion_result = detector.update(frame)
        event_manager.update(motion_result.detected, ts)

        if motion_result.detected:
            draw_motion_boxes(frame, motion_result.boxes)

            # Save occasional debug frames
            if frame_idx % 30 == 0 and DEBUG_VISUALIZE:
                out_path = f"outputs/debug_frames/frame_{frame_idx:06d}.jpg"
                cv2.imwrite(out_path, frame)

        frame_idx += 1

    reader.release()

if __name__ == "__main__":
    main()
