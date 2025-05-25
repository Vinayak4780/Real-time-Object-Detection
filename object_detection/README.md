# Real-time Object Detection System

This project implements a real-time object detection system using YOLOv8, capable of processing webcam input and detecting common objects.

## Features

- Real-time object detection from webcam feed
- Pre-trained YOLOv8 model with support for 80+ object classes
- Bounding box visualization with confidence scores
- Performance metrics tracking (FPS, processing time)
- Video recording capability for demos
- Model checkpoint saving
- Performance evaluation tools

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/object-detection.git
   cd object-detection
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running Real-time Detection

Run the detection script directly:

```bash
python real_time_detection.py --conf 0.5 --camera 0
```

Or use the provided batch file:

```bash
run_detection.bat
```

### Recording a Demo

Record a demo video while running detection:

```bash
python real_time_detection.py --conf 0.5 --camera 0 --record
```

Or use the provided batch file:

```bash
record_demo.bat
```

### Evaluating Performance

Evaluate the model's performance on a recorded video:

```bash
python evaluate_performance.py --video path/to/video.mp4 --model yolov8n.pt
```

## Command-line Options

### Real-time Detection

- `--model`: Path to the YOLO model file (default: yolov8n.pt)
- `--conf`: Confidence threshold for detections (default: 0.5)
- `--camera`: Camera device ID (default: 0)
- `--record`: Enable video recording
- `--no-save-model`: Disable model checkpoint saving

### Performance Evaluation

- `--model`: Path to the YOLO model file (default: yolov8n.pt)
- `--video`: Path to the video file to evaluate
- `--conf`: Confidence threshold for detections (default: 0.5)

## Performance

The system is optimized for real-time performance with the following achievements:

- **Average FPS**: >15 FPS on moderate hardware (depends on GPU availability)
- **Detection classes**: 80+ object categories (COCO dataset classes)
- **Confidence threshold**: Adjustable, default 0.5

## Project Structure

- `real_time_detection.py`: Main detection script
- `evaluate_performance.py`: Script for evaluating model performance
- `run_detection.bat`: Batch file to run detection
- `record_demo.bat`: Batch file to record a demo
- `requirements.txt`: List of dependencies
- `output/`: Directory for saving results and model checkpoints
- `recordings/`: Directory for saving recorded videos

## Supported Object Classes

The YOLOv8 model can detect 80+ object classes including:

Person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, TV, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush.

## Model Saving

The system automatically saves model checkpoints during operation:
- Every 100 frames during real-time detection
- At the end of a detection session
- Checkpoints are saved to the `output` directory with timestamp

## Screenshots

(Demo screenshots would be added here after running the system)

## License

This project is licensed under the MIT License - see the LICENSE file for details.