import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO
import argparse
import os
from datetime import datetime

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """Initialize the Object Detector with a YOLOv8 model."""
        self.confidence_threshold = confidence_threshold
        print(f"Loading model from {model_path}...")
        self.model = YOLO(model_path)
        print("Model loaded successfully!")
        
        # Get the model's class names
        self.class_names = self.model.names
        
        # Create output directory for saving models and results
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

    def process_frame(self, frame):
        """Process a frame and return the frame with bounding boxes."""
        start_time = time.time()
        
        # Perform detection
        results = self.model(frame, conf=self.confidence_threshold)
        
        # Visualization
        annotated_frame = results[0].plot()
        
        # Calculate and display FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_frame, results[0], fps

    def save_model_checkpoint(self):
        """Save the current model state."""
        checkpoint_path = os.path.join(self.output_dir, f"model_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
        try:
            # Save the model
            self.model.save(checkpoint_path)
            print(f"Model checkpoint saved to: {checkpoint_path}")
            return checkpoint_path
        except Exception as e:
            print(f"Error saving model: {e}")
            return None

    def run_webcam_detection(self, camera_id=0, save_video=False, save_model=True):
        """Run real-time object detection on webcam feed."""
        # Open the webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open webcam with ID {camera_id}")
            return
        
        # Get webcam properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Set up video writer if save_video is True
        video_writer = None
        if save_video:
            video_path = os.path.join(self.output_dir, f"detection_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            video_writer = cv2.VideoWriter(video_path, 
                                         cv2.VideoWriter_fourcc(*'mp4v'), 
                                         fps, 
                                         (frame_width, frame_height))
            print(f"Recording video to: {video_path}")
        
        # Performance metrics
        frame_count = 0
        total_fps = 0
        start_time = time.time()
        
        try:
            while True:
                # Read a frame from the webcam
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                # Process the frame
                annotated_frame, results, current_fps = self.process_frame(frame)
                
                # Update performance metrics
                frame_count += 1
                total_fps += current_fps
                
                # Display the frame
                cv2.imshow("Real-time Object Detection", annotated_frame)
                
                # Save the frame if recording
                if video_writer is not None:
                    video_writer.write(annotated_frame)
                
                # Exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Save model every 100 frames if requested
                if save_model and frame_count % 100 == 0:
                    saved_path = self.save_model_checkpoint()
                    if saved_path:
                        print(f"Model saved at frame {frame_count} to {saved_path}")
        
        except KeyboardInterrupt:
            print("Detection stopped by user")
        finally:
            # Calculate and display performance metrics
            elapsed_time = time.time() - start_time
            avg_fps = total_fps / frame_count if frame_count > 0 else 0
            
            print("\nPerformance Statistics:")
            print(f"Total frames processed: {frame_count}")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Total runtime: {elapsed_time:.2f} seconds")
            
            # Save final model checkpoint
            if save_model:
                saved_path = self.save_model_checkpoint()
                if saved_path:
                    print(f"Final model checkpoint saved to {saved_path}")
            
            # Release resources
            cap.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()
            
            return {
                'avg_fps': avg_fps,
                'total_frames': frame_count,
                'runtime': elapsed_time
            }

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Real-time Object Detection")
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to the YOLO model')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--record', action='store_true', help='Record a demo video')
    parser.add_argument('--no-save-model', dest='save_model', action='store_false', help='Do not save model checkpoints')
    parser.set_defaults(save_model=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create and run the object detector
    detector = ObjectDetector(model_path=args.model, confidence_threshold=args.conf)
    
    # Test model saving functionality
    if args.save_model:
        print("\nTesting model saving functionality...")
        saved_path = detector.save_model_checkpoint()
        if saved_path:
            print(f"Model saved successfully to {saved_path}")
        else:
            print("Failed to save model")
    
    # Run webcam detection if requested
    performance = detector.run_webcam_detection(
        camera_id=args.camera,
        save_video=args.record,
        save_model=args.save_model
    )
    
    print("\nDetection complete!")