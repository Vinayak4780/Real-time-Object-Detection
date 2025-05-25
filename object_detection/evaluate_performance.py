import time
import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from ultralytics import YOLO
import psutil

class PerformanceEvaluator:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """Initialize the Performance Evaluator."""
        print(f"Loading model from {model_path}...")
        self.model = YOLO(model_path)
        print("Model loaded successfully!")
        
        self.confidence_threshold = confidence_threshold
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate_video(self, video_path, save_results=True):
        """Evaluate model performance on a video file."""
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return None
            
        print(f"Evaluating performance on video: {video_path}")
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None
            
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Performance metrics
        frame_count = 0
        processing_times = []
        fps_values = []
        cpu_usage = []
        memory_usage = []
        
        # Class detection counts
        class_counts = {}
        
        try:
            while True:
                # Read a frame from the video
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Monitor system resources
                cpu_usage.append(psutil.cpu_percent())
                memory_usage.append(psutil.virtual_memory().percent)
                
                # Process the frame and measure performance
                start_time = time.time()
                results = self.model(frame, conf=self.confidence_threshold)
                end_time = time.time()
                
                # Calculate processing time and FPS
                processing_time = end_time - start_time
                processing_times.append(processing_time)
                fps_values.append(1.0 / processing_time)
                
                # Count detections by class
                for det in results[0].boxes.data:
                    cls_id = int(det[5])
                    cls_name = self.model.names[cls_id]
                    if cls_name not in class_counts:
                        class_counts[cls_name] = 0
                    class_counts[cls_name] += 1
                
                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")
                
        except KeyboardInterrupt:
            print("Evaluation stopped by user")
        finally:
            cap.release()
        
        # Calculate metrics
        avg_fps = np.mean(fps_values)
        avg_processing_time = np.mean(processing_times)
        avg_cpu_usage = np.mean(cpu_usage)
        avg_memory_usage = np.mean(memory_usage)
        
        # Prepare results
        results = {
            'avg_fps': avg_fps,
            'max_fps': np.max(fps_values),
            'min_fps': np.min(fps_values),
            'avg_processing_time': avg_processing_time,
            'total_frames': frame_count,
            'avg_cpu_usage': avg_cpu_usage,
            'avg_memory_usage': avg_memory_usage,
            'detected_classes': class_counts
        }
        
        print("\nPerformance Results:")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average processing time: {avg_processing_time*1000:.2f} ms")
        print(f"Total frames processed: {frame_count}")
        print(f"Average CPU usage: {avg_cpu_usage:.2f}%")
        print(f"Average memory usage: {avg_memory_usage:.2f}%")
        print("\nDetected classes:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} detections")
        
        # Save results if requested
        if save_results:
            self._save_performance_plots(fps_values, cpu_usage, memory_usage, class_counts)
            
        return results
    
    def _save_performance_plots(self, fps_values, cpu_usage, memory_usage, class_counts):
        """Save performance plots."""
        # Create figure for performance metrics
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        # Plot FPS over time
        axs[0].plot(fps_values)
        axs[0].set_title('FPS over time')
        axs[0].set_xlabel('Frame')
        axs[0].set_ylabel('FPS')
        axs[0].grid(True)
        
        # Plot CPU usage
        axs[1].plot(cpu_usage)
        axs[1].set_title('CPU Usage over time')
        axs[1].set_xlabel('Frame')
        axs[1].set_ylabel('CPU Usage (%)')
        axs[1].grid(True)
        
        # Plot Memory usage
        axs[2].plot(memory_usage)
        axs[2].set_title('Memory Usage over time')
        axs[2].set_xlabel('Frame')
        axs[2].set_ylabel('Memory Usage (%)')
        axs[2].grid(True)
        
        plt.tight_layout()
        perf_plot_path = os.path.join(self.output_dir, f"performance_metrics_{time.strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(perf_plot_path)
        print(f"Performance plots saved to: {perf_plot_path}")
        
        # Create figure for class distribution
        plt.figure(figsize=(12, 6))
        plt.bar(class_counts.keys(), class_counts.values())
        plt.title('Object Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Number of Detections')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        class_plot_path = os.path.join(self.output_dir, f"class_distribution_{time.strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(class_plot_path)
        print(f"Class distribution plot saved to: {class_plot_path}")
        
        plt.close('all')

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Object Detection Performance")
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to the YOLO model')
    parser.add_argument('--video', type=str, required=True, help='Path to the video file to evaluate')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create and run the evaluator
    evaluator = PerformanceEvaluator(model_path=args.model, confidence_threshold=args.conf)
    evaluator.evaluate_video(args.video)