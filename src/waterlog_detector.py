import cv2
import numpy as np
import json
import time
from datetime import datetime
from ultralytics import YOLO
import torch
from pathlib import Path
import argparse

class WaterlogDetector:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize the waterlog detector
        
        Args:
            model_path: Path to the YOLO model (use pre-trained or custom trained)
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.results_data = []
        
        # If using a custom model, ensure it's trained for waterlog detection
        # For demo purposes, we'll use object detection and filter for water-like objects
        
    def preprocess_frame(self, frame):
        """
        Preprocess frame for better waterlog detection
        This can include color space conversion, edge detection, etc.
        """
        # Convert to HSV for better water detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Water typically has specific HSV ranges (dark, low saturation)
        # This is a simplified approach - real implementation would need proper training data
        lower_water = np.array([0, 0, 0])
        upper_water = np.array([180, 100, 100])
        
        water_mask = cv2.inRange(hsv, lower_water, upper_water)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
        
        return frame, water_mask
    
    def detect_waterlogs(self, frame, timestamp, frame_number):
        """
        Detect waterlogs in a single frame
        
        Args:
            frame: Input video frame
            timestamp: Timestamp of the frame
            frame_number: Frame number in the video
            
        Returns:
            List of detection results
        """
        # Preprocess frame
        processed_frame, water_mask = self.preprocess_frame(frame)
        
        # Run YOLO detection
        results = self.model(processed_frame, conf=self.confidence_threshold)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Calculate center point and area
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    area = width * height
                    
                    # Check if detection overlaps with water-like regions
                    roi_mask = water_mask[int(y1):int(y2), int(x1):int(x2)]
                    if roi_mask.size > 0:
                        water_pixels = np.sum(roi_mask > 0)
                        overlap_ratio = water_pixels / (width * height) if (width * height) > 0 else 0
                        
                        # Consider it a waterlog if there's significant overlap with water-like pixels
                        # or if it's detected as a relevant class (you'd need to train for waterlog class)
                        if overlap_ratio > 0.3 or class_id in [0]:  # Adjust class_id for your model
                            detection = {
                                "timestamp": timestamp,
                                "frame_number": frame_number,
                                "detection_id": len(detections),
                                "bounding_box": {
                                    "x1": float(x1),
                                    "y1": float(y1),
                                    "x2": float(x2),
                                    "y2": float(y2),
                                    "center_x": center_x,
                                    "center_y": center_y,
                                    "width": width,
                                    "height": height
                                },
                                "area_pixels": area,
                                "confidence": float(confidence),
                                "water_overlap_ratio": float(overlap_ratio),
                                "class_name": "waterlog"
                            }
                            detections.append(detection)
        
        return detections
    
    def process_video(self, video_path, output_json_path=None, save_annotated_video=False, output_video_path=None):
        """
        Process entire video and detect waterlogs
        
        Args:
            video_path: Path to input video
            output_json_path: Path to save JSON results
            save_annotated_video: Whether to save video with annotations
            output_video_path: Path for annotated video output
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Total frames: {total_frames}, Resolution: {frame_width}x{frame_height}")
        
        # Setup video writer if saving annotated video
        if save_annotated_video and output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        frame_number = 0
        self.results_data = []
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate timestamp
            timestamp = frame_number / fps
            datetime_str = datetime.fromtimestamp(time.time() + timestamp).isoformat()
            
            # Detect waterlogs
            detections = self.detect_waterlogs(frame, timestamp, frame_number)
            
            if detections:
                frame_data = {
                    "frame_number": frame_number,
                    "timestamp_seconds": timestamp,
                    "datetime": datetime_str,
                    "detections": detections
                }
                self.results_data.append(frame_data)
                
                # Draw annotations if saving video
                if save_annotated_video:
                    for detection in detections:
                        bbox = detection["bounding_box"]
                        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        
                        # Draw confidence and info
                        label = f"Waterlog: {detection['confidence']:.2f}"
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        # Draw center point
                        center = (bbox["center_x"], bbox["center_y"])
                        cv2.circle(frame, center, 5, (0, 0, 255), -1)
            
            # Write frame if saving video
            if save_annotated_video and output_video_path:
                out.write(frame)
            
            frame_number += 1
            
            # Progress update
            if frame_number % 100 == 0:
                progress = (frame_number / total_frames) * 100
                elapsed_time = time.time() - start_time
                eta = (elapsed_time / frame_number) * (total_frames - frame_number)
                print(f"Progress: {progress:.1f}% ({frame_number}/{total_frames}) - ETA: {eta:.1f}s")
        
        # Cleanup
        cap.release()
        if save_annotated_video and output_video_path:
            out.release()
        
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Total waterlog detections: {sum(len(frame['detections']) for frame in self.results_data)}")
        
        # Save results to JSON
        if output_json_path:
            self.save_results_json(output_json_path)
        
        return self.results_data
    
    def save_results_json(self, output_path):
        """Save detection results to JSON file"""
        results_summary = {
            "video_info": {
                "total_frames_with_detections": len(self.results_data),
                "total_detections": sum(len(frame['detections']) for frame in self.results_data),
                "processing_timestamp": datetime.now().isoformat()
            },
            "detections_by_frame": self.results_data
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"Results saved to: {output_path}")
    
    def get_detection_statistics(self):
        """Get statistics about detections"""
        if not self.results_data:
            return {}
        
        total_detections = sum(len(frame['detections']) for frame in self.results_data)
        frames_with_detections = len(self.results_data)
        
        # Calculate average confidence
        all_confidences = []
        all_areas = []
        
        for frame in self.results_data:
            for detection in frame['detections']:
                all_confidences.append(detection['confidence'])
                all_areas.append(detection['area_pixels'])
        
        stats = {
            "total_detections": total_detections,
            "frames_with_detections": frames_with_detections,
            "average_confidence": np.mean(all_confidences) if all_confidences else 0,
            "average_area": np.mean(all_areas) if all_areas else 0,
            "confidence_range": {
                "min": float(np.min(all_confidences)) if all_confidences else 0,
                "max": float(np.max(all_confidences)) if all_confidences else 0
            }
        }
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='Waterlog Detection using YOLO')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to YOLO model')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output-json', type=str, help='Path to save JSON results')
    parser.add_argument('--output-video', type=str, help='Path to save annotated video')
    parser.add_argument('--save-video', action='store_true', help='Save annotated video')
    
    args = parser.parse_args()
    
    # Create detector
    detector = WaterlogDetector(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    # Set default output paths if not provided
    video_path = Path(args.video)
    if not args.output_json:
        args.output_json = video_path.stem + '_waterlog_detections.json'
    
    if args.save_video and not args.output_video:
        args.output_video = video_path.stem + '_annotated.mp4'
    
    try:
        # Process video
        results = detector.process_video(
            video_path=args.video,
            output_json_path=args.output_json,
            save_annotated_video=args.save_video,
            output_video_path=args.output_video
        )
        
        # Print statistics
        stats = detector.get_detection_statistics()
        print("\n=== Detection Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        print(f"\nResults saved to: {args.output_json}")
        if args.save_video:
            print(f"Annotated video saved to: {args.output_video}")
            
    except Exception as e:
        print(f"Error processing video: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
