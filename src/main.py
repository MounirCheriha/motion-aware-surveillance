import logging
from pipeline import VideoProcessorPipeline

ENABLE_LABELING = True 
YOLO_FRAME_STRIDE = 5  # run YOLO every N frames


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    video_path = "data/one-by-one-person-detection.mp4"
    pipeline = VideoProcessorPipeline(
        video_path=video_path,
        enable_labeling=ENABLE_LABELING,
        yolo_stride=YOLO_FRAME_STRIDE
    )
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
