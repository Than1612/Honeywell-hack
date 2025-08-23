import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
import json
from pathlib import Path
import requests
import zipfile
import shutil
from sklearn.model_selection import train_test_split

class DatasetHandler:
    """
    Handler for surveillance datasets (Avenue, UCSD) with actual training integration
    """
    
    def __init__(self, data_dir: str = "data", output_dir: str = "processed_data"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Dataset information and download URLs
        self.datasets = {
            'avenue': {
                'name': 'Avenue Dataset',
                'description': 'Avenue dataset for anomaly detection in surveillance videos',
                'url': 'http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html',
                'local_dir': 'avenue',
                'video_dir': 'videos',
                'annotation_file': 'annotations.txt',
                'frame_rate': 25,
                'resolution': (640, 360),
                'train_videos': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15'],
                'test_videos': ['16', '17', '18', '19', '20', '21']
            },
            'ucsd': {
                'name': 'UCSD Anomaly Detection Dataset',
                'description': 'UCSD pedestrian dataset for anomaly detection',
                'url': 'http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm',
                'local_dir': 'ucsd',
                'video_dir': 'videos',
                'annotation_file': 'annotations.txt',
                'frame_rate': 10,
                'resolution': (238, 158),
                'train_videos': ['Train001', 'Train002', 'Train003', 'Train004', 'Train005', 'Train006', 'Train007', 'Train008'],
                'test_videos': ['Test001', 'Test002', 'Test003', 'Test004', 'Test005', 'Test006', 'Test007', 'Test008', 'Test009', 'Test010']
            }
        }
        
        logging.info("Dataset handler initialized")
    
    def download_avenue_dataset(self, force_download: bool = False) -> bool:
        """
        Download Avenue dataset from CUHK
        Note: This is a placeholder - actual download requires manual steps due to dataset access
        """
        dataset_info = self.datasets['avenue']
        local_path = os.path.join(self.data_dir, dataset_info['local_dir'])
        
        if os.path.exists(local_path) and not force_download:
            logging.info(f"Avenue dataset already exists at {local_path}")
            return True
        
        logging.info("Avenue dataset download instructions:")
        logging.info("1. Visit: http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html")
        logging.info("2. Download the Avenue dataset")
        logging.info("3. Extract to: " + local_path)
        logging.info("4. Ensure videos are in: " + os.path.join(local_path, dataset_info['video_dir']))
        
        # Create directory structure
        os.makedirs(local_path, exist_ok=True)
        os.makedirs(os.path.join(local_path, dataset_info['video_dir']), exist_ok=True)
        
        # Create placeholder for manual download
        readme_path = os.path.join(local_path, "README.txt")
        with open(readme_path, 'w') as f:
            f.write("Avenue Dataset - Manual Download Required\n")
            f.write("==========================================\n\n")
            f.write("1. Visit: http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html\n")
            f.write("2. Download the Avenue dataset\n")
            f.write("3. Extract videos to the 'videos' folder\n")
            f.write("4. Run: python -m data.dataset_handler --prepare-avenue\n")
        
        return False
    
    def download_ucsd_dataset(self, force_download: bool = False) -> bool:
        """
        Download UCSD dataset
        Note: This is a placeholder - actual download requires manual steps
        """
        dataset_info = self.datasets['ucsd']
        local_path = os.path.join(self.data_dir, dataset_info['local_dir'])
        
        if os.path.exists(local_path) and not force_download:
            logging.info(f"UCSD dataset already exists at {local_path}")
            return True
        
        logging.info("UCSD dataset download instructions:")
        logging.info("1. Visit: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm")
        logging.info("2. Download the UCSD Anomaly Detection Dataset")
        logging.info("3. Extract to: " + local_path)
        
        # Create directory structure
        os.makedirs(local_path, exist_ok=True)
        os.makedirs(os.path.join(local_path, dataset_info['video_dir']), exist_ok=True)
        
        # Create placeholder for manual download
        readme_path = os.path.join(local_path, "README.txt")
        with open(readme_path, 'w') as f:
            f.write("UCSD Anomaly Detection Dataset - Manual Download Required\n")
            f.write("========================================================\n\n")
            f.write("1. Visit: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm\n")
            f.write("2. Download the UCSD Anomaly Detection Dataset\n")
            f.write("3. Extract videos to the 'videos' folder\n")
            f.write("4. Run: python -m data.dataset_handler --prepare-ucsd\n")
        
        return False
    
    def prepare_avenue_dataset(self) -> bool:
        """Prepare Avenue dataset for training"""
        dataset_info = self.datasets['avenue']
        local_path = os.path.join(self.data_dir, dataset_info['local_dir'])
        video_path = os.path.join(local_path, dataset_info['video_dir'])
        
        if not os.path.exists(video_path):
            logging.error(f"Video directory not found: {video_path}")
            return False
        
        # Check if videos exist
        video_files = [f for f in os.listdir(video_path) if f.endswith(('.avi', '.mp4', '.mov'))]
        if not video_files:
            logging.error(f"No video files found in {video_path}")
            return False
        
        logging.info(f"Found {len(video_files)} video files")
        
        # Create training structure
        train_dir = os.path.join(self.output_dir, 'avenue', 'train')
        val_dir = os.path.join(self.output_dir, 'avenue', 'val')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Process videos and extract frames
        for video_file in video_files:
            video_path_full = os.path.join(video_path, video_file)
            video_name = os.path.splitext(video_file)[0]
            
            # Determine if this is train or test video
            if video_name in dataset_info['train_videos']:
                output_dir = os.path.join(train_dir, video_name)
            else:
                output_dir = os.path.join(val_dir, video_name)
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract frames
            self._extract_video_frames(video_path_full, output_dir, frame_interval=5)
        
        # Create YOLO format annotations
        self._create_yolo_annotations('avenue')
        
        logging.info("Avenue dataset prepared successfully")
        return True
    
    def prepare_ucsd_dataset(self) -> bool:
        """Prepare UCSD dataset for training"""
        dataset_info = self.datasets['ucsd']
        local_path = os.path.join(self.data_dir, dataset_info['local_dir'])
        video_path = os.path.join(local_path, dataset_info['video_dir'])
        
        if not os.path.exists(video_path):
            logging.error(f"Video directory not found: {video_path}")
            return False
        
        # Check if videos exist
        video_files = [f for f in os.listdir(video_path) if f.endswith(('.avi', '.mp4', '.mov'))]
        if not video_files:
            logging.error(f"No video files found in {video_path}")
            return False
        
        logging.info(f"Found {len(video_files)} video files")
        
        # Create training structure
        train_dir = os.path.join(self.output_dir, 'ucsd', 'train')
        val_dir = os.path.join(self.output_dir, 'ucsd', 'val')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Process videos and extract frames
        for video_file in video_files:
            video_path_full = os.path.join(video_path, video_file)
            video_name = os.path.splitext(video_file)[0]
            
            # Determine if this is train or validation video
            if video_name in dataset_info['train_videos']:
                output_dir = os.path.join(train_dir, video_name)
            else:
                output_dir = os.path.join(val_dir, video_name)
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract frames
            self._extract_video_frames(video_path_full, output_dir, frame_interval=3)
        
        # Create YOLO format annotations
        self._create_yolo_annotations('ucsd')
        
        logging.info("UCSD dataset prepared successfully")
        return True
    
    def _extract_video_frames(self, video_path: str, output_dir: str, frame_interval: int = 1):
        """Extract frames from video at specified interval"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Could not open video: {video_path}")
            return
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        logging.info(f"Extracted {saved_count} frames from {video_path}")
    
    def _create_yolo_annotations(self, dataset_name: str):
        """Create YOLO format annotations for training"""
        dataset_info = self.datasets[dataset_name]
        output_base = os.path.join(self.output_dir, dataset_name)
        
        # Create YOLO dataset structure
        yolo_dir = os.path.join(output_base, 'yolo_dataset')
        os.makedirs(yolo_dir, exist_ok=True)
        
        # Create directories
        images_train = os.path.join(yolo_dir, 'images', 'train')
        images_val = os.path.join(yolo_dir, 'images', 'val')
        labels_train = os.path.join(yolo_dir, 'labels', 'train')
        labels_val = os.path.join(yolo_dir, 'labels', 'val')
        
        for dir_path in [images_train, images_val, labels_train, labels_val]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Create dataset.yaml
        dataset_yaml = os.path.join(yolo_dir, 'dataset.yaml')
        with open(dataset_yaml, 'w') as f:
            f.write(f"# {dataset_info['name']} Dataset Configuration\n")
            f.write(f"path: {os.path.abspath(yolo_dir)}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("nc: 1\n")  # Number of classes (person)
            f.write("names: ['person']\n")
        
        # Copy images and create labels
        self._prepare_yolo_data(output_base, yolo_dir, dataset_name)
        
        logging.info(f"YOLO dataset created at {yolo_dir}")
    
    def _prepare_yolo_data(self, source_base: str, yolo_dir: str, dataset_name: str):
        """Prepare YOLO format data"""
        # Copy images
        train_source = os.path.join(source_base, 'train')
        val_source = os.path.join(source_base, 'val')
        
        # Copy training images
        self._copy_images_to_yolo(train_source, os.path.join(yolo_dir, 'images', 'train'))
        self._copy_images_to_yolo(val_source, os.path.join(yolo_dir, 'images', 'val'))
        
        # Create labels (simplified - in real scenario, you'd have actual annotations)
        self._create_synthetic_labels(yolo_dir, dataset_name)
    
    def _copy_images_to_yolo(self, source_dir: str, target_dir: str):
        """Copy images to YOLO directory structure"""
        if not os.path.exists(source_dir):
            return
        
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.jpg'):
                    # Create unique filename
                    unique_name = f"{os.path.basename(root)}_{file}"
                    target_path = os.path.join(target_dir, unique_name)
                    
                    # Copy file
                    source_path = os.path.join(root, file)
                    shutil.copy2(source_path, target_path)
    
    def _create_synthetic_labels(self, yolo_dir: str, dataset_name: str):
        """Create synthetic labels for training (in real scenario, use actual annotations)"""
        # This is a simplified approach - in production, you'd use actual annotations
        
        # For training images
        train_images = os.path.join(yolo_dir, 'images', 'train')
        train_labels = os.path.join(yolo_dir, 'labels', 'train')
        
        for img_file in os.listdir(train_images):
            if img_file.endswith('.jpg'):
                # Create corresponding label file
                label_file = img_file.replace('.jpg', '.txt')
                label_path = os.path.join(train_labels, label_file)
                
                # Create synthetic label (person in center of image)
                # Format: class_id center_x center_y width height (normalized)
                with open(label_path, 'w') as f:
                    f.write("0 0.5 0.5 0.3 0.6\n")  # Person class, center position
        
        # For validation images
        val_images = os.path.join(yolo_dir, 'images', 'val')
        val_labels = os.path.join(yolo_dir, 'labels', 'val')
        
        for img_file in os.listdir(val_images):
            if img_file.endswith('.jpg'):
                label_file = img_file.replace('.jpg', '.txt')
                label_path = os.path.join(val_labels, label_file)
                
                with open(label_path, 'w') as f:
                    f.write("0 0.5 0.5 0.3 0.6\n")
    
    def get_yolo_dataset_path(self, dataset_name: str) -> Optional[str]:
        """Get path to prepared YOLO dataset"""
        yolo_path = os.path.join(self.output_dir, dataset_name, 'yolo_dataset')
        if os.path.exists(yolo_path):
            return yolo_path
        return None
    
    def create_training_script(self, dataset_name: str):
        """Create YOLOv8 training script for the dataset"""
        dataset_info = self.datasets[dataset_name]
        yolo_path = self.get_yolo_dataset_path(dataset_name)
        
        if not yolo_path:
            logging.error(f"YOLO dataset not found for {dataset_name}")
            return
        
        # Create training script
        script_path = os.path.join(self.output_dir, f'train_{dataset_name}_yolov8.py')
        
        script_content = f'''#!/usr/bin/env python3
"""
YOLOv8 Training Script for {dataset_info['name']}
"""

from ultralytics import YOLO
import os

def train_yolov8():
    # Load a model
    model = YOLO('yolov8n.pt')  # load pretrained model (recommended for training)
    
    # Train the model
    results = model.train(
        data='{os.path.join(yolo_path, "dataset.yaml")}',
        epochs=100,
        imgsz=640,
        batch=16,
        name='yolov8_{dataset_name}_custom'
    )
    
    # Validate the model
    results = model.val()
    
    # Export the model
    model.export(format='onnx')
    
    print(f"Training completed for {dataset_name} dataset!")

if __name__ == '__main__':
    train_yolov8()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        logging.info(f"Training script created: {script_path}")
        return script_path

def main():
    """Main function for dataset handling"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset Handler for AI Surveillance System')
    parser.add_argument('--data-dir', '-d', default='data',
                       help='Directory containing datasets')
    parser.add_argument('--output-dir', '-o', default='processed_data',
                       help='Output directory for processed data')
    parser.add_argument('--download-avenue', action='store_true',
                       help='Download Avenue dataset')
    parser.add_argument('--download-ucsd', action='store_true',
                       help='Download UCSD dataset')
    parser.add_argument('--prepare-avenue', action='store_true',
                       help='Prepare Avenue dataset for training')
    parser.add_argument('--prepare-ucsd', action='store_true',
                       help='Prepare UCSD dataset for training')
    parser.add_argument('--create-training-scripts', action='store_true',
                       help='Create YOLOv8 training scripts')
    
    args = parser.parse_args()
    
    # Create handler
    handler = DatasetHandler(args.data_dir, args.output_dir)
    
    if args.download_avenue:
        handler.download_avenue_dataset()
    
    if args.download_ucsd:
        handler.download_ucsd_dataset()
    
    if args.prepare_avenue:
        if handler.prepare_avenue_dataset():
            print("Avenue dataset prepared successfully!")
        else:
            print("Failed to prepare Avenue dataset")
    
    if args.prepare_ucsd:
        if handler.prepare_ucsd_dataset():
            print("UCSD dataset prepared successfully!")
        else:
            print("Failed to prepare UCSD dataset")
    
    if args.create_training_scripts:
        handler.create_training_script('avenue')
        handler.create_training_script('ucsd')
        print("Training scripts created!")

if __name__ == '__main__':
    main()
