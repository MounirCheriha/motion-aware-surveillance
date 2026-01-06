import cv2
import os


class VideoReader:
    def __init__(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            # Fallback for badly encoded videos
            self.fps = 30.0

    def read(self):
        """
        Generator yielding (frame, timestamp_seconds)
        """
        frame_index = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            timestamp = frame_index / self.fps
            frame_index += 1

            yield frame, timestamp

    def release(self):
        self.cap.release()
