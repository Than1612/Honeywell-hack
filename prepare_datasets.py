#!/usr/bin/env python3
"""
Dataset Preparation Script for AI Surveillance System
Downloads and prepares Avenue and UCSD datasets for YOLOv8 training
"""

import os
import sys
import logging
import requests
import zipfile
import shutil
from pathlib import Path
import argparse
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DatasetPreparator:
    """
    Prepares surveillance datasets for training
    """
    
    def __init__(self, data_dir: str = "data", output_dir: str = "processed_data"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Dataset information
        self.datasets = {
            'avenue': {
                'name': 'Avenue Dataset',
                'description': 'Avenue dataset for anomaly detection in surveillance videos',
                'download_url': 'https://www.dropbox.com/s/3dw5ue5v7bfrgu6/Avenue_Dataset.zip',
                'local_dir': 'avenue',
                'expected_size_mb': 500,
                'train_videos': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15'],
                'test_videos': ['16', '17', '18', '19', '20', '21']
            },
            'ucsd': {
                'name': 'UCSD Anomaly Detection Dataset',
                'description': 'UCSD pedestrian dataset for anomaly detection',
                'download_url': 'https://www.dropbox.com/s/3dw5ue5v7bfrgu6/UCSD_Anomaly_Dataset.zip',
                'local_dir': 'ucsd',
                'expected_size_mb': 300,
                'train_videos': ['Train001', 'Train002', 'Train003', 'Train004', 'Train005', 'Train006', 'Train007', 'Train008'],
                'test_videos': ['Test001', 'Test002', 'Test003', 'Test004', 'Test005', 'Test006', 'Test007', 'Test008', 'Test009', 'Test010']
            }
        }
    
    def download_dataset(self, dataset_id: str, force_download: bool = False) -> bool:
        """Download dataset from source"""
        if dataset_id not in self.datasets:
            logging.error(f"Unknown dataset: {dataset_id}")
            return False
        
        dataset_info = self.datasets[dataset_id]
        local_path = os.path.join(self.data_dir, dataset_info['local_dir'])
        zip_path = os.path.join(self.data_dir, f"{dataset_id}.zip")
        
        # Check if already exists
        if os.path.exists(local_path) and not force_download:
            logging.info(f"Dataset {dataset_id} already exists at {local_path}")
            return True
        
        logging.info(f"Downloading {dataset_info['name']}...")
        logging.info(f"URL: {dataset_info['download_url']}")
        
        try:
            # Download file
            response = requests.get(dataset_info['download_url'], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            logging.info(f"Downloaded: {downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB ({percent:.1f}%)")
            
            logging.info(f"Download completed: {zip_path}")
            
            # Extract dataset
            return self._extract_dataset(dataset_id, zip_path)
            
        except Exception as e:
            logging.error(f"Download failed: {e}")
            return False
    
    def _extract_dataset(self, dataset_id: str, zip_path: str) -> bool:
        """Extract downloaded dataset"""
        dataset_info = self.datasets[dataset_id]
        local_path = os.path.join(self.data_dir, dataset_info['local_dir'])
        
        logging.info(f"Extracting {dataset_id} dataset...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            # Clean up zip file
            os.remove(zip_path)
            
            logging.info(f"Dataset extracted to: {local_path}")
            return True
            
        except Exception as e:
            logging.error(f"Extraction failed: {e}")
            return False
    
    def prepare_yolo_dataset(self, dataset_id: str) -> bool:
        """Prepare dataset in YOLO format for training"""
        if dataset_id not in self.datasets:
            logging.error(f"Unknown dataset: {dataset_id}")
            return False
        
        dataset_info = self.datasets[dataset_id]
        local_path = os.path.join(self.data_dir, dataset_info['local_dir'])
        
        if not os.path.exists(local_path):
            logging.error(f"Dataset not found: {local_path}")
            return False
        
        logging.info(f"Preparing YOLO dataset for {dataset_id}...")
        
        # Create YOLO directory structure
        yolo_dir = os.path.join(self.output_dir, dataset_id, 'yolo_dataset')
        train_images = os.path.join(yolo_dir, 'images', 'train')
        train_labels = os.path.join(yolo_dir, 'labels', 'train')
        val_images = os.path.join(yolo_dir, 'images', 'val')
        val_labels = os.path.join(yolo_dir, 'labels', 'val')
        
        for dir_path in [train_images, train_labels, val_images, val_labels]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Process videos and extract frames
        video_files = self._find_video_files(local_path)
        
        if not video_files:
            logging.error(f"No video files found in {local_path}")
            return False
        
        logging.info(f"Found {len(video_files)} video files")
        
        # Process each video
        for video_file in video_files:
            video_path = os.path.join(local_path, video_file)
            video_name = os.path.splitext(video_file)[0]
            
            # Determine if this is train or test video
            if video_name in dataset_info['train_videos']:
                output_images = train_images
                output_labels = train_labels
            else:
                output_images = val_images
                output_labels = val_labels
            
            # Extract frames
            self._extract_video_frames(video_path, output_images, video_name)
            
            # Create labels (simplified - in real scenario, use actual annotations)
            self._create_synthetic_labels(output_labels, video_name)
        
        # Create dataset.yaml
        self._create_dataset_yaml(yolo_dir, dataset_id)
        
        logging.info(f"YOLO dataset prepared: {yolo_dir}")
        return True
    
    def _find_video_files(self, directory: str) -> list:
        """Find video files in directory"""
        video_extensions = ['.avi', '.mp4', '.mov', '.mkv', '.flv']
        video_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(file)
        
        return video_files
    
    def _extract_video_frames(self, video_path: str, output_dir: str, video_name: str, frame_interval: int = 5):
        """Extract frames from video"""
        import cv2
        
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
                frame_path = os.path.join(output_dir, f"{video_name}_frame_{saved_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        logging.info(f"Extracted {saved_count} frames from {video_name}")
    
    def _create_synthetic_labels(self, labels_dir: str, video_name: str):
        """Create synthetic labels for training (in real scenario, use actual annotations)"""
        # This creates simplified labels - in production, use actual annotations from the datasets
        
        # Find corresponding images
        images_dir = labels_dir.replace('labels', 'images')
        if not os.path.exists(images_dir):
            return
        
        image_files = [f for f in os.listdir(images_dir) if f.startswith(video_name) and f.endswith('.jpg')]
        
        for image_file in image_files:
            # Create corresponding label file
            label_file = image_file.replace('.jpg', '.txt')
            label_path = os.path.join(labels_dir, label_file)
            
            # Create synthetic label (person in center of image)
            # Format: class_id center_x center_y width height (normalized)
            # Class 0 = person
            with open(label_path, 'w') as f:
                f.write("0 0.5 0.5 0.3 0.6\n")  # Person class, center position
    
    def _create_dataset_yaml(self, yolo_dir: str, dataset_id: str):
        """Create dataset.yaml file for YOLOv8 training"""
        dataset_info = self.datasets[dataset_id]
        
        yaml_content = {
            'path': os.path.abspath(yolo_dir),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,  # Number of classes (person)
            'names': ['person']
        }
        
        yaml_path = os.path.join(yolo_dir, 'dataset.yaml')
        
        # Write YAML file
        with open(yaml_path, 'w') as f:
            f.write(f"# {dataset_info['name']} Dataset Configuration\n")
            f.write(f"path: {yaml_content['path']}\n")
            f.write(f"train: {yaml_content['train']}\n")
            f.write(f"val: {yaml_content['val']}\n")
            f.write(f"nc: {yaml_content['nc']}\n")
            f.write(f"names: {yaml_content['names']}\n")
        
        logging.info(f"Dataset YAML created: {yaml_path}")
    
    def create_training_script(self, dataset_id: str):
        """Create training script for the dataset"""
        script_content = f'''#!/usr/bin/env python3
"""
Training script for {dataset_id} dataset
"""

from ultralytics import YOLO
import os

def train_yolov8():
    # Dataset path
    dataset_yaml = "processed_data/{dataset_id}/yolo_dataset/dataset.yaml"
    
    if not os.path.exists(dataset_yaml):
        print(f"Dataset not found: {{dataset_yaml}}")
        print("Please run prepare_datasets.py first")
        return
    
    # Load model
    model = YOLO('yolov8n.pt')  # load pretrained model
    
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        name=f'yolov8_{dataset_id}_custom'
    )
    
    print(f"Training completed for {dataset_id} dataset!")

if __name__ == '__main__':
    train_yolov8()
'''
        
        script_path = os.path.join(self.output_dir, f'train_{dataset_id}.py')
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        logging.info(f"Training script created: {script_path}")
        return script_path
    
    def run_quick_test(self, dataset_id: str):
        """Run a quick test to verify dataset preparation"""
        yolo_dir = os.path.join(self.output_dir, dataset_id, 'yolo_dataset')
        
        if not os.path.exists(yolo_dir):
            logging.error(f"YOLO dataset not found: {yolo_dir}")
            return False
        
        # Check structure
        required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
        for dir_path in required_dirs:
            full_path = os.path.join(yolo_dir, dir_path)
            if not os.path.exists(full_path):
                logging.error(f"Missing directory: {full_path}")
                return False
        
        # Count files
        train_images = len(os.listdir(os.path.join(yolo_dir, 'images', 'train')))
        val_images = len(os.listdir(os.path.join(yolo_dir, 'images', 'val')))
        train_labels = len(os.listdir(os.path.join(yolo_dir, 'labels', 'train')))
        val_labels = len(os.listdir(os.path.join(yolo_dir, 'labels', 'val')))
        
        logging.info(f"Dataset verification for {dataset_id}:")
        logging.info(f"  Training images: {train_images}")
        logging.info(f"  Validation images: {val_images}")
        logging.info(f"  Training labels: {train_labels}")
        logging.info(f"  Validation labels: {val_labels}")
        
        return train_images > 0 and val_images > 0

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Dataset Preparation for AI Surveillance System')
    parser.add_argument('--dataset', '-d', choices=['avenue', 'ucsd', 'both'], default='both',
                       help='Dataset to prepare')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='processed_data', help='Output directory')
    parser.add_argument('--download', action='store_true', help='Download datasets')
    parser.add_argument('--prepare', action='store_true', help='Prepare YOLO datasets')
    parser.add_argument('--test', action='store_true', help='Test dataset preparation')
    parser.add_argument('--create-scripts', action='store_true', help='Create training scripts')
    parser.add_argument('--force', action='store_true', help='Force re-download/preparation')
    
    args = parser.parse_args()
    
    # Create preparator
    preparator = DatasetPreparator(args.data_dir, args.output_dir)
    
    # Determine datasets to process
    if args.dataset == 'both':
        datasets_to_process = ['avenue', 'ucsd']
    else:
        datasets_to_process = [args.dataset]
    
    # Process each dataset
    for dataset_id in datasets_to_process:
        logging.info(f"\n{'='*50}")
        logging.info(f"Processing {dataset_id} dataset")
        logging.info(f"{'='*50}")
        
        try:
            # Download if requested
            if args.download:
                if not preparator.download_dataset(dataset_id, args.force):
                    logging.error(f"Failed to download {dataset_id} dataset")
                    continue
            
            # Prepare if requested
            if args.prepare:
                if not preparator.prepare_yolo_dataset(dataset_id):
                    logging.error(f"Failed to prepare {dataset_id} dataset")
                    continue
            
            # Test if requested
            if args.test:
                if not preparator.run_quick_test(dataset_id):
                    logging.error(f"Dataset test failed for {dataset_id}")
                    continue
            
            # Create training script if requested
            if args.create_scripts:
                preparator.create_training_script(dataset_id)
            
            logging.info(f"✓ {dataset_id} dataset processed successfully!")
            
        except Exception as e:
            logging.error(f"Error processing {dataset_id} dataset: {e}")
            continue
    
    logging.info(f"\n{'='*50}")
    logging.info("Dataset preparation completed!")
    logging.info(f"Data directory: {args.data_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"{'='*50}")

if __name__ == '__main__':
    main()
