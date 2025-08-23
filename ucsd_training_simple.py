#!/usr/bin/env python3
"""
UCSD Training Pipeline - HACKATHON VERSION
Simple and fast training for your hackathon project
"""

import os
import sys
import logging
from pathlib import Path
import json
import time
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleUCSDTrainer:
    """Simple trainer for hackathon - just get it working!"""
    
    def __init__(self, dataset_root: str = "UCSD_Anomaly_Dataset.v1p2"):
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path("models")
        self.output_dir.mkdir(exist_ok=True)
        
        # Simple training configs
        self.configs = {
            'quick': {'epochs': 10, 'batch': 8},
            'fast': {'epochs': 25, 'batch': 16},
            'full': {'epochs': 50, 'batch': 16}
        }
        
        logger.info(f"Simple UCSD Trainer ready! Dataset: {self.dataset_root}")
    
    def train_model(self, mode: str = 'quick', model_size: str = 'n') -> Dict:
        """Train a model - hackathon style!"""
        try:
            logger.info(f"🚀 Starting {mode} training with YOLOv8{model_size}")
            
            # Step 1: Prepare dataset
            logger.info("📁 Preparing dataset...")
            from ucsd_dataset_handler import UCSDDatasetHandler
            handler = UCSDDatasetHandler(str(self.dataset_root))
            
            if mode == 'quick':
                dataset_path = handler.create_quick_test_dataset()
            else:
                dataset_path = handler.prepare_yolo_dataset()
            
            logger.info(f"✅ Dataset ready: {dataset_path}")
            
            # Step 2: Train model
            logger.info("🎯 Training model...")
            from ultralytics import YOLO
            
            # Load base model
            base_model = f"yolov8{model_size}.pt"
            model = YOLO(base_model)
            
            # Get config
            config = self.configs[mode]
            
            # Train!
            results = model.train(
                data=dataset_path,
                epochs=config['epochs'],
                imgsz=640,
                batch=config['batch'],
                device='cpu',  # Use CPU for hackathon
                project=str(self.output_dir),
                name=f'yolov8{model_size}_ucsd_{mode}',
                exist_ok=True,
                verbose=True
            )
            
            # Get model path
            model_name = f'yolov8{model_size}_ucsd_{mode}'
            best_model_path = self.output_dir / model_name / 'weights' / 'best.pt'
            
            if not best_model_path.exists():
                best_model_path = self.output_dir / model_name / 'weights' / 'last.pt'
            
            logger.info(f"🎉 Training complete! Model saved at: {best_model_path}")
            
            # Step 3: Quick export
            logger.info("📤 Exporting model...")
            onnx_path = model.export(format='onnx')
            logger.info(f"✅ ONNX exported: {onnx_path}")
            
            return {
                'status': 'success',
                'mode': mode,
                'model_size': model_size,
                'model_path': str(best_model_path),
                'onnx_path': str(onnx_path),
                'epochs': config['epochs'],
                'message': f'Model trained successfully in {mode} mode!'
            }
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def list_models(self) -> Dict:
        """List available trained models"""
        try:
            models = list(self.output_dir.glob("yolov8*_ucsd_*"))
            
            available = []
            for model_dir in models:
                if model_dir.is_dir():
                    weights_dir = model_dir / 'weights'
                    if weights_dir.exists():
                        best_model = weights_dir / 'best.pt'
                        last_model = weights_dir / 'last.pt'
                        
                        if best_model.exists() or last_model.exists():
                            model_info = {
                                'name': model_dir.name,
                                'best_model': str(best_model) if best_model.exists() else None,
                                'last_model': str(last_model) if last_model.exists() else None,
                                'created': time.ctime(model_dir.stat().st_ctime)
                            }
                            available.append(model_info)
            
            return {
                'status': 'success',
                'models': available,
                'count': len(available)
            }
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    def test_model(self, model_path: str) -> Dict:
        """Quick test of a trained model"""
        try:
            from ultralytics import YOLO
            
            model = YOLO(model_path)
            
            # Test with dummy image
            import numpy as np
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Run inference
            results = model(test_image, verbose=False)
            
            return {
                'status': 'success',
                'model_loaded': True,
                'inference_test': 'passed',
                'message': 'Model working correctly!'
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }


def main():
    """Main function - hackathon style!"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple UCSD Trainer - Hackathon Version')
    parser.add_argument('--mode', choices=['quick', 'fast', 'full'], 
                       default='quick', help='Training mode')
    parser.add_argument('--model-size', choices=['n', 's', 'm'], 
                       default='n', help='Model size')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--test', type=str, help='Test a specific model')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SimpleUCSDTrainer()
    
    try:
        if args.list:
            # List models
            models = trainer.list_models()
            print("\n" + "="*50)
            print("AVAILABLE MODELS")
            print("="*50)
            
            if models['status'] == 'success':
                for model in models['models']:
                    print(f"📁 {model['name']}")
                    print(f"   Best: {model['best_model']}")
                    print(f"   Created: {model['created']}")
                    print()
            else:
                print("❌ No models found")
                
        elif args.test:
            # Test model
            print(f"🧪 Testing model: {args.test}")
            result = trainer.test_model(args.test)
            
            if result['status'] == 'success':
                print("✅ Model test passed!")
            else:
                print(f"❌ Model test failed: {result['error']}")
                
        else:
            # Train model
            print(f"🚀 Starting {args.mode} training...")
            result = trainer.train_model(args.mode, args.model_size)
            
            if result['status'] == 'success':
                print("\n" + "🎉 TRAINING SUCCESSFUL! 🎉")
                print("="*50)
                print(f"Mode: {result['mode']}")
                print(f"Model: {result['model_size']}")
                print(f"Model saved: {result['model_path']}")
                print(f"ONNX exported: {result['onnx_path']}")
                print(f"Epochs: {result['epochs']}")
                print("="*50)
                print("Your model is ready for the hackathon! 🚀")
            else:
                print(f"❌ Training failed: {result['error']}")
                
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
