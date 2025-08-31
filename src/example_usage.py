#!/usr/bin/env python3
"""
Example usage script for waterlog detection system
"""

import os
import json
from pathlib import Path
from waterlog_detector import WaterlogDetector

def example_basic_detection():
    """Basic waterlog detection example"""
    print("=== Basic Waterlog Detection Example ===")
    
    # Initialize detector
    detector = WaterlogDetector(
        model_path='yolov8n.pt',  # Use pre-trained model initially
        confidence_threshold=0.5
    )
    
    # Example video path (replace with your video)
    video_path = "sample_road_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        print("Please provide a valid video file path")
        return
    
    # Process video
    results = detector.process_video(
        video_path=video_path,
        output_json_path="waterlog_detections.json",
        save_annotated_video=True,
        output_video_path="annotated_output.mp4"
    )
    
    # Print statistics
    stats = detector.get_detection_statistics()
    print(f"\nDetection Results:")
    print(f"- Total detections: {stats['total_detections']}")
    print(f"- Frames with detections: {stats['frames_with_detections']}")
    print(f"- Average confidence: {stats['average_confidence']:.2f}")
    print(f"- Average detection area: {stats['average_area']:.0f} pixels")
    
    return results

def example_custom_model():
    """Example using custom trained model"""
    print("\n=== Custom Model Example ===")
    
    custom_model_path = "waterlog_models/waterlog_detection/weights/best.pt"
    
    if not os.path.exists(custom_model_path):
        print(f"Custom model not found: {custom_model_path}")
        print("Train a custom model first using train_waterlog_yolo.py")
        return
    
    # Initialize with custom model
    detector = WaterlogDetector(
        model_path=custom_model_path,
        confidence_threshold=0.3  # Lower threshold for custom model
    )
    
    video_path = "test_video.mp4"
    results = detector.process_video(video_path, "custom_model_results.json")
    
    return results

def example_real_time_processing():
    """Example of real-time processing"""
    print("\n=== Real-time Processing Example ===")
    
    import cv2
    
    detector = WaterlogDetector(confidence_threshold=0.6)
    
    # Use webcam (0) or video file
    cap = cv2.VideoCapture(0)  # Change to video path for file input
    
    if not cap.isOpened():
        print("Error opening video stream")
        return
    
    frame_number = 0
    fps = 30  # Assume 30 FPS
    
    print("Press 'q' to quit real-time detection")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_number / fps
            
            # Detect waterlogs in current frame
            detections = detector.detect_waterlogs(frame, timestamp, frame_number)
            
            # Draw detections on frame
            for detection in detections:
                bbox = detection["bounding_box"]
                x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                # Draw label
                label = f"Waterlog: {detection['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                print(f"Frame {frame_number}: Waterlog detected with confidence {detection['confidence']:.2f}")
            
            # Display frame
            cv2.imshow('Waterlog Detection', frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_number += 1
            
    finally:
        cap.release()
        cv2.destroyAllWindows()

def example_batch_processing():
    """Example of processing multiple videos"""
    print("\n=== Batch Processing Example ===")
    
    video_directory = "input_videos"
    output_directory = "detection_results"
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    detector = WaterlogDetector(confidence_threshold=0.5)
    
    # Process all videos in directory
    video_files = list(Path(video_directory).glob("*.mp4"))
    
    if not video_files:
        print(f"No video files found in {video_directory}")
        return
    
    results_summary = []
    
    for video_path in video_files:
        print(f"Processing: {video_path.name}")
        
        output_json = Path(output_directory) / f"{video_path.stem}_detections.json"
        output_video = Path(output_directory) / f"{video_path.stem}_annotated.mp4"
        
        try:
            results = detector.process_video(
                video_path=str(video_path),
                output_json_path=str(output_json),
                save_annotated_video=True,
                output_video_path=str(output_video)
            )
            
            stats = detector.get_detection_statistics()
            
            video_summary = {
                "video_name": video_path.name,
                "status": "success",
                "total_detections": stats['total_detections'],
                "frames_with_detections": stats['frames_with_detections'],
                "average_confidence": stats['average_confidence'],
                "output_files": {
                    "json": str(output_json),
                    "annotated_video": str(output_video)
                }
            }
            
        except Exception as e:
            video_summary = {
                "video_name": video_path.name,
                "status": "error",
                "error_message": str(e)
            }
            print(f"Error processing {video_path.name}: {e}")
        
        results_summary.append(video_summary)
    
    # Save batch processing summary
    summary_path = Path(output_directory) / "batch_processing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"Batch processing complete. Summary saved to: {summary_path}")
    return results_summary

def example_configure_detection_parameters():
    """Example of configuring detection parameters"""
    print("\n=== Detection Parameter Configuration ===")
    
    # Different confidence thresholds for different scenarios
    scenarios = {
        "high_precision": {"confidence": 0.8, "description": "Fewer false positives"},
        "balanced": {"confidence": 0.5, "description": "Balance between precision and recall"},
        "high_recall": {"confidence": 0.2, "description": "Detect more waterlogs, may include false positives"}
    }
    
    video_path = "test_video.mp4"
    
    for scenario_name, config in scenarios.items():
        print(f"\n{scenario_name.upper()} Configuration:")
        print(f"Description: {config['description']}")
        print(f"Confidence threshold: {config['confidence']}")
        
        detector = WaterlogDetector(confidence_threshold=config['confidence'])
        
        if os.path.exists(video_path):
            results = detector.process_video(
                video_path=video_path,
                output_json_path=f"{scenario_name}_results.json"
            )
            
            stats = detector.get_detection_statistics()
            print(f"Results: {stats['total_detections']} detections")

def create_config_file():
    """Create a configuration file for the system"""
    config = {
        "model_settings": {
            "default_model": "yolov8n.pt",
            "custom_model_path": "waterlog_models/waterlog_detection/weights/best.pt",
            "confidence_threshold": 0.5,
            "supported_models": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
        },
        "preprocessing": {
            "enable_hsv_filtering": True,
            "hsv_lower": [0, 0, 0],
            "hsv_upper": [180, 100, 100],
            "morphology_kernel_size": 5,
            "water_overlap_threshold": 0.3
        },
        "output_settings": {
            "default_json_output": "waterlog_detections.json",
            "save_annotated_video": False,
            "annotation_color": [0, 255, 255],
            "annotation_thickness": 2
        },
        "performance": {
            "batch_size": 1,
            "use_gpu": True,
            "max_video_resolution": [1920, 1080],
            "process_every_nth_frame": 1
        },
        "training": {
            "default_epochs": 100,
            "default_batch_size": 16,
            "default_image_size": 640,
            "dataset_split": {"train": 0.7, "val": 0.2, "test": 0.1}
        }
    }
    
    config_path = "waterlog_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration file created: {config_path}")
    return config

def main():
    """Main function to run examples"""
    print("Waterlog Detection System Examples")
    print("=" * 50)
    
    # Create configuration file
    config = create_config_file()
    
    # Run examples
    examples = [
        ("Basic Detection", example_basic_detection),
        ("Custom Model", example_custom_model),
        ("Real-time Processing", example_real_time_processing),
        ("Batch Processing", example_batch_processing),
        ("Parameter Configuration", example_configure_detection_parameters)
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    
    print("0. Run all examples")
    
    try:
        choice = input("\nSelect example to run (0-{}): ".format(len(examples)))
        choice = int(choice)
        
        if choice == 0:
            # Run all examples
            for name, func in examples:
                try:
                    print(f"\n{'='*50}")
                    print(f"Running: {name}")
                    print('='*50)
                    func()
                except Exception as e:
                    print(f"Error in {name}: {e}")
        elif 1 <= choice <= len(examples):
            name, func = examples[choice - 1]
            print(f"\nRunning: {name}")
            func()
        else:
            print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
