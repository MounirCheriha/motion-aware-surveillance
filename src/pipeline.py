import logging
from video.reader import VideoReader
from motion.motion_detector import MotionDetector
from events.event_manager import EventManager
from video.writer import EventVideoWriter
from events.metadata_writer import MetadataWriter


class VideoProcessorPipeline:
    def __init__(self, video_path, enable_labeling, yolo_stride):
        
        self.enable_labeling = enable_labeling
        self.yolo_stride = yolo_stride

        self.reader = VideoReader(video_path)
        self.writer = EventVideoWriter(
            output_dir="outputs/events",
            fps=self.reader.fps
        )

        self.event_manager = EventManager(inactivity_timeout=2.0)   
        self.metadata_writer = MetadataWriter(
            output_path="outputs/metadata/events.json"
        )

        self.motion_detector = MotionDetector(min_area=800)
        self.object_detector = None
        if self.enable_labeling:
            from detection.object_detector import ObjectDetector
            self.object_detector = ObjectDetector(
                model_name="yolov8n.pt",
                conf_threshold=0.5,
                roi_padding=10,
                enabled=self.enable_labeling
            )
    
    def run_pipeline(self):

        frame_count = 0
        for frame, ts in self.reader.read():
            motion_result = self.motion_detector.update(frame)
            signal = self.event_manager.update(motion_result.motion_detected, ts)

            if signal == "start":
                self.writer.start(self.event_manager.event_id, frame.shape)
                frame_count = 0

            if self.event_manager.event_active:
                # Run YOLO only during active events
                if self.enable_labeling and (frame_count % self.yolo_stride == 0):
                    detections = self.object_detector.detect_on_rois(
                        frame,
                        motion_result.bounding_boxes
                    )
                    for d in detections:
                        x1, y1, x2, y2 = d["bbox"]
                        label = d["label"]
                        conf = d["confidence"]

                    self.event_manager.add_detections([d["label"] for d in detections])
            
                self.writer.write(frame)
                frame_count += 1
            
            if signal == "end":
                clip_path = self.writer.stop()
                event_data = self.event_manager.get_event_metadata(ts)
                event_data["clip_path"] = clip_path

                self.metadata_writer.add_event(event_data)
                self.event_manager.reset()
                logging.info(f"Event {event_data['event_id']} saved")   

        self.metadata_writer.save()
        self.reader.release()    
        return