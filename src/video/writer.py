import cv2
import os


class EventVideoWriter:
    def __init__(self, output_dir: str, fps: float):
        self.output_dir = output_dir
        self.fps = fps
        self.writer = None
        self.current_path = None

        os.makedirs(output_dir, exist_ok=True)

    def start(self, event_id: int, frame_shape):
        """
        Start writing a new event video.
        """
        height, width, _ = frame_shape
        filename = f"event_{event_id:03d}.mp4"
        self.current_path = os.path.join(self.output_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.current_path, fourcc, self.fps, (width, height)
        )

        if not self.writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter")

    def write(self, frame):
        if self.writer is not None:
            self.writer.write(frame)

    def stop(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            return self.current_path

        return None
