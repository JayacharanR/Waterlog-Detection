#!/usr/bin/env python3
"""
Custom YOLO Training Script for Waterlog Detection

This script trains a YOLO model specifically for detecting waterlogs on roads.
You'll need to prepare a dataset with annotated waterlog images.

Dataset structure should be:
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml

"""

import os
import yaml
from ultralytics import YOLO
from pathlib import Path
import argparse
import shutil

def create_dataset_yaml(dataset_path, class_names=['waterlog']):
    """
    Create the data.yaml file required for YOLO training
    
    Args:
        dataset_path: Path to the dataset directory
        class_names: List of class names
    """
    data_yaml = {
        'path': str(dataset_path),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Created data.yaml at: {yaml_path}")
    return yaml_path

def prepare_dataset_structure(dataset_path):
    """
    Create the required dataset directory structure
    """
    dataset_path = Path(dataset_path)
    
    # Create directories
    dirs_to_create = [
        'images/train',
        'images/val', 
        'images/test',
        'labels/train',
        'labels/val',
        'labels/test'
    ]
    
    for dir_name in dirs_to_create:
        dir_path = dataset_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Dataset structure created at: {dataset_path}")
    return dataset_path

def train_waterlog_model(dataset_path, model_size='yolov8n.pt', epochs=100, img_size=640, batch_size=16):
    """
    Train YOLO model for waterlog detection
    
    Args:
        dataset_path: Path to dataset with data.yaml
        model_size: YOLO model size ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt')
        epochs: Number of training epochs
        img_size: Image size for training
        batch_size: Batch size for training
    """
    
    # Load pre-trained YOLO model
    model = YOLO(model_size)
    
    # Find data.yaml file
    data_yaml = os.path.join(dataset_path, 'data.yaml')
    if not os.path.exists(data_yaml):
        print(f"Creating data.yaml file...")
        data_yaml = create_dataset_yaml(dataset_path)
    
    print(f"Starting training with:")
    print(f"  - Dataset: {dataset_path}")
    print(f"  - Model: {model_size}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Image size: {img_size}")
    print(f"  - Batch size: {batch_size}")
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name='waterlog_detection',
        project='waterlog_models',
        save=True,
        plots=True,
        device=0 if os.system('nvidia-smi') == 0 else 'cpu'  # Use GPU if available
    )
    
    print("Training completed!")
    print(f"Best model saved at: {results.save_dir}")
    
    return results

def validate_model(model_path, dataset_path):
    """
    Validate the trained model
    """
    model = YOLO(model_path)
    
    data_yaml = os.path.join(dataset_path, 'data.yaml')
    
    # Validate the model
    results = model.val(
        data=data_yaml,
        split='test'
    )
    
    print("Validation Results:")
    print(f"mAP50: {results.box.map50:.3f}")
    print(f"mAP50-95: {results.box.map:.3f}")
    
    return results

def create_sample_annotation_script():
    """
    Create a sample script for converting annotations to YOLO format
    """
    script_content = '''#!/usr/bin/env python3
"""
Sample script to convert annotations to YOLO format
Modify this according to your annotation format
"""

import os
import json
from pathlib import Path

def convert_bbox_to_yolo(img_width, img_height, bbox):
    """
    Convert bounding box to YOLO format
    
    Args:
        img_width: Image width
        img_height: Image height  
        bbox: Bounding box in format [x1, y1, x2, y2]
    
    Returns:
        YOLO format: [class_id, center_x, center_y, width, height] (normalized)
    """
    x1, y1, x2, y2 = bbox
    
    # Calculate center point and dimensions
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    
    # Normalize to image dimensions
    center_x /= img_width
    center_y /= img_height
    width /= img_width
    height /= img_height
    
    return [0, center_x, center_y, width, height]  # 0 is class_id for waterlog

def convert_annotations(annotations_dir, images_dir, output_dir):
    """
    Convert your annotation format to YOLO format
    Modify this function based on your annotation format
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Example: If you have JSON annotations
    for annotation_file in Path(annotations_dir).glob("*.json"):
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # Extract image info and annotations
        image_name = data['image_name']  # Modify according to your format
        img_width = data['width']        # Modify according to your format
        img_height = data['height']      # Modify according to your format
        
        # Create YOLO annotation file
        txt_filename = annotation_file.stem + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)
        
        with open(txt_path, 'w') as f:
            for annotation in data['annotations']:  # Modify according to your format
                bbox = annotation['bbox']  # [x1, y1, x2, y2]
                yolo_bbox = convert_bbox_to_yolo(img_width, img_height, bbox)
                f.write(' '.join(map(str, yolo_bbox)) + '\\n')

if __name__ == "__main__":
    # Modify these paths according to your setup
    annotations_dir = "path/to/your/annotations"
    images_dir = "path/to/your/images"
    output_dir = "path/to/yolo/labels"
    
    convert_annotations(annotations_dir, images_dir, output_dir)
'''
    
    with open('convert_annotations.py', 'w') as f:
        f.write(script_content)
    
    print("Sample annotation conversion script created: convert_annotations.py")

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model for waterlog detection')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--setup-only', action='store_true', help='Only setup dataset structure')
    parser.add_argument('--validate', type=str, help='Path to trained model for validation')
    
    args = parser.parse_args()
    
    # Setup dataset structure
    if args.setup_only:
        prepare_dataset_structure(args.dataset)
        create_dataset_yaml(args.dataset)
        create_sample_annotation_script()
        print("Dataset structure setup complete!")
        print("Next steps:")
        print("1. Add your images to the images/ subdirectories")
        print("2. Add corresponding YOLO format labels to the labels/ subdirectories")
        print("3. Modify convert_annotations.py if needed")
        print("4. Run training with: python train_waterlog_yolo.py --dataset your_dataset_path")
        return
    
    # Validate existing model
    if args.validate:
        validate_model(args.validate, args.dataset)
        return
    
    # Train model
    try:
        results = train_waterlog_model(
            dataset_path=args.dataset,
            model_size=args.model,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch_size
        )
        
        print("Training completed successfully!")
        print(f"Model saved in: waterlog_models/")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
'''