#!/usr/bin/env python3
"""
YOLOv8 Training Script for AI Surveillance System
Uses Avenue and UCSD datasets for actual training
"""

import os
import sys
import logging
from pathlib import Path
import yaml
from ultralytics import YOLO
import argparse
from typing import Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class YOLOv8Trainer:
    """
    YOLOv8 trainer for surveillance datasets
    """
    
    def __init__(self, data_dir: str = "data", output_dir: str = "models"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            'avenue': {
                'name': 'Avenue Dataset',
                'path': os.path.join(data_dir, 'avenue', 'yolo_dataset'),
                'yaml': 'dataset.yaml',
                'classes': ['person'],
                'num_classes': 1
            },
            'ucsd': {
                'name': 'UCSD Dataset',
                'path': os.path.join(data_dir, 'ucsd', 'yolo_dataset'),
                'yaml': 'dataset.yaml',
                'classes': ['person'],
                'num_classes': 1
            }
        }
        
        # Training configurations
        self.training_configs = {
            'avenue': {
                'epochs': 100,
                'batch_size': 16,
                'img_size': 640,
                'learning_rate': 0.01,
                'patience': 20
            },
            'ucsd': {
                'epochs': 80,
                'batch_size': 16,
                'img_size': 640,
                'learning_rate': 0.01,
                'patience': 15
            }
        }
    
    def check_dataset_availability(self) -> Dict[str, bool]:
        """Check which datasets are available"""
        available = {}
        
        for dataset_id, dataset_info in self.datasets.items():
            dataset_path = dataset_info['path']
            yaml_path = os.path.join(dataset_path, dataset_info['yaml'])
            
            if os.path.exists(dataset_path) and os.path.exists(yaml_path):
                available[dataset_id] = True
                logging.info(f"✓ {dataset_info['name']} available at {dataset_path}")
            else:
                available[dataset_id] = False
                logging.warning(f"✗ {dataset_info['name']} not found at {dataset_path}")
        
        return available
    
    def prepare_dataset_yaml(self, dataset_id: str) -> str:
        """Prepare dataset.yaml file for YOLOv8 training"""
        dataset_info = self.datasets[dataset_id]
        dataset_path = dataset_info['path']
        yaml_path = os.path.join(dataset_path, dataset_info['yaml'])
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Create dataset.yaml content
        yaml_content = {
            'path': os.path.abspath(dataset_path),
            'train': 'images/train',
            'val': 'images/val',
            'nc': dataset_info['num_classes'],
            'names': dataset_info['classes']
        }
        
        # Write dataset.yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        logging.info(f"Dataset YAML created: {yaml_path}")
        return yaml_path
    
    def train_model(self, dataset_id: str, model_size: str = 'n', 
                   pretrained_weights: str = None) -> str:
        """
        Train YOLOv8 model on specified dataset
        
        Args:
            dataset_id: Dataset to use ('avenue' or 'ucsd')
            model_size: Model size ('n', 's', 'm', 'l', 'x')
            pretrained_weights: Path to pretrained weights (optional)
        
        Returns:
            Path to trained model
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_id}")
        
        dataset_info = self.datasets[dataset_id]
        training_config = self.training_configs[dataset_id]
        
        logging.info(f"Starting training on {dataset_info['name']}")
        logging.info(f"Model size: YOLOv8{model_size}")
        
        # Prepare dataset
        yaml_path = self.prepare_dataset_yaml(dataset_id)
        
        # Load model
        if pretrained_weights and os.path.exists(pretrained_weights):
            model = YOLO(pretrained_weights)
            logging.info(f"Loaded pretrained weights: {pretrained_weights}")
        else:
            model = YOLO(f'yolov8{model_size}.pt')
            logging.info(f"Loaded base model: yolov8{model_size}.pt")
        
        # Training parameters
        train_params = {
            'data': yaml_path,
            'epochs': training_config['epochs'],
            'imgsz': training_config['img_size'],
            'batch': training_config['batch_size'],
            'lr0': training_config['learning_rate'],
            'patience': training_config['patience'],
            'save': True,
            'save_period': 10,
            'cache': False,
            'device': 'auto',
            'workers': 8,
            'project': self.output_dir,
            'name': f'yolov8{model_size}_{dataset_id}_custom',
            'exist_ok': True
        }
        
        logging.info("Training parameters:")
        for key, value in train_params.items():
            logging.info(f"  {key}: {value}")
        
        # Start training
        try:
            logging.info("Starting training...")
            results = model.train(**train_params)
            
            # Save training results
            results_path = os.path.join(self.output_dir, f'yolov8{model_size}_{dataset_id}_custom', 'results.yaml')
            with open(results_path, 'w') as f:
                yaml.dump(results, f, default_flow_style=False)
            
            logging.info(f"Training completed successfully!")
            logging.info(f"Results saved to: {results_path}")
            
            return results_path
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise
    
    def validate_model(self, model_path: str, dataset_id: str):
        """Validate trained model on dataset"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logging.info(f"Validating model: {model_path}")
        
        # Load model
        model = YOLO(model_path)
        
        # Prepare dataset
        yaml_path = self.prepare_dataset_yaml(dataset_id)
        
        # Run validation
        results = model.val(data=yaml_path)
        
        logging.info("Validation completed!")
        logging.info(f"mAP50: {results.box.map50:.4f}")
        logging.info(f"mAP50-95: {results.box.map:.4f}")
        
        return results
    
    def export_model(self, model_path: str, export_format: str = 'onnx'):
        """Export model to different formats"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logging.info(f"Exporting model to {export_format} format...")
        
        # Load model
        model = YOLO(model_path)
        
        # Export
        exported_path = model.export(format=export_format)
        
        logging.info(f"Model exported to: {exported_path}")
        return exported_path
    
    def create_inference_script(self, model_path: str, dataset_id: str):
        """Create inference script for trained model"""
        script_content = f'''#!/usr/bin/env python3
"""
Inference script for trained YOLOv8 model on {dataset_id} dataset
"""

from ultralytics import YOLO
import cv2
import numpy as np
import argparse

def run_inference(video_path: str, model_path: str = "{model_path}"):
    # Load trained model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {{video_path}}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model(frame, verbose=False)
        
        # Draw results
        annotated_frame = results[0].plot()
        
        # Display
        cv2.imshow('Inference', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv8 Inference on {dataset_id} dataset')
    parser.add_argument('--video', '-v', required=True, help='Path to video file')
    parser.add_argument('--model', '-m', default='{model_path}', help='Path to trained model')
    
    args = parser.parse_args()
    run_inference(args.video, args.model)
'''
        
        script_path = os.path.join(self.output_dir, f'inference_{dataset_id}.py')
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        logging.info(f"Inference script created: {script_path}")
        return script_path

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='YOLOv8 Training for Surveillance System')
    parser.add_argument('--dataset', '-d', choices=['avenue', 'ucsd', 'both'], default='avenue',
                       help='Dataset to train on')
    parser.add_argument('--model-size', '-s', choices=['n', 's', 'm', 'l', 'x'], default='n',
                       help='YOLOv8 model size')
    parser.add_argument('--pretrained', '-p', help='Path to pretrained weights')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='models', help='Output directory')
    parser.add_argument('--validate', action='store_true', help='Validate after training')
    parser.add_argument('--export', action='store_true', help='Export model after training')
    parser.add_argument('--create-scripts', action='store_true', help='Create inference scripts')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = YOLOv8Trainer(args.data_dir, args.output_dir)
    
    # Check dataset availability
    available_datasets = trainer.check_dataset_availability()
    
    if not any(available_datasets.values()):
        logging.error("No datasets available! Please prepare datasets first.")
        logging.info("Run: python -m data.dataset_handler --prepare-avenue --prepare-ucsd")
        sys.exit(1)
    
    # Training datasets
    if args.dataset == 'both':
        datasets_to_train = [d for d in ['avenue', 'ucsd'] if available_datasets[d]]
    else:
        datasets_to_train = [args.dataset] if available_datasets[args.dataset] else []
    
    if not datasets_to_train:
        logging.error(f"Dataset {args.dataset} not available!")
        sys.exit(1)
    
    # Train on each dataset
    trained_models = []
    
    for dataset_id in datasets_to_train:
        try:
            logging.info(f"\n{'='*50}")
            logging.info(f"Training on {dataset_id} dataset")
            logging.info(f"{'='*50}")
            
            # Train model
            results_path = trainer.train_model(
                dataset_id, 
                args.model_size, 
                args.pretrained
            )
            
            trained_models.append((dataset_id, results_path))
            
            # Validate if requested
            if args.validate:
                trainer.validate_model(results_path, dataset_id)
            
            # Export if requested
            if args.export:
                trainer.export_model(results_path)
            
            # Create inference script if requested
            if args.create_scripts:
                trainer.create_inference_script(results_path, dataset_id)
            
        except Exception as e:
            logging.error(f"Training failed for {dataset_id}: {e}")
            continue
    
    # Summary
    if trained_models:
        logging.info(f"\n{'='*50}")
        logging.info("Training Summary")
        logging.info(f"{'='*50}")
        for dataset_id, model_path in trained_models:
            logging.info(f"✓ {dataset_id}: {model_path}")
        logging.info(f"\nModels saved to: {args.output_dir}")
    else:
        logging.error("No models were trained successfully!")

if __name__ == '__main__':
    main()
