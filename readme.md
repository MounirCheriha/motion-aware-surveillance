# Motion-Aware Surveillance System with Event-Based Object Detection

This project implements a **motion-aware video surveillance pipeline** that detects movement in a video stream, segments motion into discrete events, optionally classifies the moving objects (e.g., person or animal), and stores both **event clips** and **structured metadata**.

The system is designed to be:
- **Efficient** (motion-first, ML second)
- **Headless-friendly** (server / WSL compatible)
- **Modular and extensible**
- Suitable for **live surveillance use cases**

---

## Project Motivation

Continuous object detection on surveillance streams is computationally expensive and often unnecessary.  
Most frames are static.

This project follows a **real-world surveillance design pattern**:

> Use cheap classical computer vision to detect *when* something happens, and run machine learning only to determine *what* happened.

---

## High-Level Architecture

```

Video Stream / File
↓
Motion Detection (OpenCV)
↓
Event Segmentation
↓
Event-Based Video Recording
↓
Optional YOLO Object Detection
↓
Event Metadata Export (JSON)

```
---

## Key Features (Implemented)

### 1. Motion Detection (Classical Computer Vision)
- Background subtraction using OpenCV (MOG2)
- Noise reduction via morphology
- Bounding boxes around motion regions
- Designed for fixed surveillance cameras

### 2. Event Segmentation
- Motion signals are grouped into **events**
- Events start when motion appears
- Events end after configurable inactivity timeout
- Short events can be discarded automatically

### 3. Event-Based Video Recording
- Video clips are recorded **only during events**
- One clip per event
- Clean lifecycle management of video writers
- Suitable for long-running streams

### 4. Optional Object Detection (YOLOv8)
- Uses pretrained YOLOv8 (`yolov8n`)
- Runs **only during active events**
- Can be fully disabled via configuration
- Labeling is throttled (not every frame)
- Event-level label aggregation (e.g., `person`, `animal`)

### 5. Structured Metadata Export
- Each event produces a JSON record:
  - Start time
  - End time
  - Duration
  - Detected labels
  - Clip path
- Metadata is saved as a single JSON file for downstream use

---

## Example Event Metadata

```json
{
  "event_id": 1,
  "start_time": 4.12,
  "end_time": 9.87,
  "duration": 5.75,
  "labels": ["person"],
  "clip_path": "outputs/events/event_001.mp4"
}
```
---

## Project Structure

```
motion-aware-surveillance/
│
├── src/
│   ├── video/
│   │   ├── reader.py        # Video input abstraction
│   │   └── writer.py        # Event-based video recording
│   │
│   ├── motion/
│   │   ├── motion_detector.py
│   │   └── motion_result.py
│   │
│   ├── detection/
│   │   └── object_detector.py
│   │
│   ├── events/
│   │   ├── event_manager.py
│   │   └── metadata_writer.py
│   │
│   └── main.py              # Pipeline orchestration
│
├── configs/
│   └── default.yaml
│
├── outputs/
│   ├── events/
│   └── metadata/
│
├── requirements.txt
└── README.md
```

---

## Running the Project

### 1. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
python3 src/main.py
```

The first run will automatically download the YOLO model if labeling is enabled.

---

## Configuration Options

Key runtime options (currently defined in code, planned to move fully to YAML):

* Enable / disable object labeling
* YOLO confidence threshold
* YOLO inference frame stride
* Motion area threshold
* Event inactivity timeout

This allows the system to run in:

* **Motion-only mode**
* **Motion + labeling mode**

---

## Design Decisions

* **Headless execution**: No GUI dependencies (suitable for servers and WSL)
* **Motion-first strategy**: Reduces unnecessary ML inference
* **Event-level semantics**: Labels describe events, not individual frames
* **Modular components**: Easy to swap detectors or video sources

---

## Planned Next Steps

The following improvements are intentionally left for future iterations:

### 1. Performance Optimization

* Motion-guided YOLO inference on Regions of Interest (ROIs)
* Further reduction of YOLO calls for long events
* Optional GPU acceleration

### 2. Live YouTube Stream Integration

* Support for private YouTube live streams as input
* Continuous monitoring of surveillance feeds
* Graceful reconnection handling

### 3. Improved Event Semantics

* Person vs animal classification logic
* Dominant label per event
* Confidence-based event labeling

### 4. Production Enhancements

* Full YAML-based configuration
* Logging instead of print statements
* Docker support for deployment
* Alerting (email / messaging)

---

## Ethical Considerations

* No face recognition
* No identity tracking
* Intended for educational purposes
* Designed for private surveillance data only

---

## License

MIT

````