#!/usr/bin/env python3
"""
Complete Training Pipeline for AI Surveillance System
Downloads Avenue and UCSD datasets, trains YOLOv8 models, and integrates them
"""

import os
import sys
import logging
import subprocess
import time
from pathlib import Path
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TrainingPipeline:
    """
    Complete training pipeline for surveillance system
    """
    
    def __init__(self, data_dir: str = "data", output_dir: str = "models"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.processed_data_dir = "processed_data"
        
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Pipeline steps
        self.pipeline_steps = [
            'download_datasets',
            'prepare_datasets', 
            'train_models',
            'validate_models',
            'export_models',
            'test_integration'
        ]
        
        logging.info("Training pipeline initialized")
    
    def run_complete_pipeline(self, datasets: list = None, model_size: str = 'n'):
        """Run the complete training pipeline"""
        if datasets is None:
            datasets = ['avenue', 'ucsd']
        
        logging.info("=" * 60)
        logging.info("Starting Complete Training Pipeline")
        logging.info("=" * 60)
        
        start_time = time.time()
        results = {}
        
        try:
            # Step 1: Download datasets
            logging.info("\nStep 1: Downloading Datasets")
            results['download'] = self.download_datasets(datasets)
            
            # Step 2: Prepare datasets for YOLO training
            logging.info("\nStep 2: Preparing Datasets")
            results['prepare'] = self.prepare_datasets(datasets)
            
            # Step 3: Train YOLOv8 models
            logging.info("\nStep 3: Training YOLOv8 Models")
            results['train'] = self.train_models(datasets, model_size)
            
            # Step 4: Validate trained models
            logging.info("\nStep 4: Validating Models")
            results['validate'] = self.validate_models(datasets)
            
            # Step 5: Export models
            logging.info("\nStep 5: Exporting Models")
            results['export'] = self.export_models(datasets)
            
            # Step 6: Test integration
            logging.info("\nStep 6: Testing Integration")
            results['integration'] = self.test_integration(datasets)
            
            # Pipeline completed
            total_time = time.time() - start_time
            logging.info(f"\nPipeline completed in {total_time/60:.1f} minutes")
            
            # Print summary
            self.print_pipeline_summary(results)
            
            return results
            
        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            return {'error': str(e)}
    
    def download_datasets(self, datasets: list) -> dict:
        """Download required datasets"""
        results = {}
        
        for dataset in datasets:
            logging.info(f"Downloading {dataset} dataset...")
            
            try:
                # Run dataset preparation script
                cmd = [
                    sys.executable, 'prepare_datasets.py',
                    '--dataset', dataset,
                    '--download',
                    '--data-dir', self.data_dir
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    results[dataset] = {'status': 'success', 'message': 'Downloaded successfully'}
                    logging.info(f"✓ {dataset} dataset downloaded")
                else:
                    results[dataset] = {'status': 'failed', 'message': result.stderr}
                    logging.error(f"✗ {dataset} dataset download failed")
                
            except Exception as e:
                results[dataset] = {'status': 'error', 'message': str(e)}
                logging.error(f"Error downloading {dataset}: {e}")
        
        return results
    
    def prepare_datasets(self, datasets: list) -> dict:
        """Prepare datasets for YOLO training"""
        results = {}
        
        for dataset in datasets:
            logging.info(f"Preparing {dataset} dataset for YOLO training...")
            
            try:
                # Run dataset preparation script
                cmd = [
                    sys.executable, 'prepare_datasets.py',
                    '--dataset', dataset,
                    '--prepare',
                    '--test',
                    '--data-dir', self.data_dir,
                    '--output-dir', self.processed_data_dir
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    results[dataset] = {'status': 'success', 'message': 'Prepared successfully'}
                    logging.info(f"✓ {dataset} dataset prepared")
                else:
                    results[dataset] = {'status': 'failed', 'message': result.stderr}
                    logging.error(f"✗ {dataset} dataset preparation failed")
                
            except Exception as e:
                results[dataset] = {'status': 'error', 'message': str(e)}
                logging.error(f"Error preparing {dataset}: {e}")
        
        return results
    
    def train_models(self, datasets: list, model_size: str) -> dict:
        """Train YOLOv8 models on datasets"""
        results = {}
        
        for dataset in datasets:
            logging.info(f"Training YOLOv8 model on {dataset} dataset...")
            
            try:
                # Run training script
                cmd = [
                    sys.executable, 'train_yolov8.py',
                    '--dataset', dataset,
                    '--model-size', model_size,
                    '--data-dir', self.data_dir,
                    '--output-dir', self.output_dir,
                    '--validate',
                    '--export',
                    '--create-scripts'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    results[dataset] = {'status': 'success', 'message': 'Training completed'}
                    logging.info(f"✓ {dataset} model training completed")
                else:
                    results[dataset] = {'status': 'failed', 'message': result.stderr}
                    logging.error(f"✗ {dataset} model training failed")
                
            except Exception as e:
                results[dataset] = {'status': 'error', 'message': str(e)}
                logging.error(f"Error training {dataset} model: {e}")
        
        return results
    
    def validate_models(self, datasets: list) -> dict:
        """Validate trained models"""
        results = {}
        
        for dataset in datasets:
            logging.info(f"Validating {dataset} model...")
            
            try:
                # Check if model exists
                model_path = os.path.join(self.output_dir, f'yolov8n_{dataset}_custom', 'weights', 'best.pt')
                
                if os.path.exists(model_path):
                    results[dataset] = {
                        'status': 'success', 
                        'message': 'Model validated',
                        'model_path': model_path
                    }
                    logging.info(f"✓ {dataset} model validated: {model_path}")
                else:
                    results[dataset] = {'status': 'failed', 'message': 'Model not found'}
                    logging.error(f"✗ {dataset} model not found")
                
            except Exception as e:
                results[dataset] = {'status': 'error', 'message': str(e)}
                logging.error(f"Error validating {dataset} model: {e}")
        
        return results
    
    def export_models(self, datasets: list) -> dict:
        """Export models to different formats"""
        results = {}
        
        for dataset in datasets:
            logging.info(f"Exporting {dataset} model...")
            
            try:
                # Check if model exists
                model_path = os.path.join(self.output_dir, f'yolov8n_{dataset}_custom', 'weights', 'best.pt')
                
                if os.path.exists(model_path):
                    # Export to ONNX format
                    from ultralytics import YOLO
                    model = YOLO(model_path)
                    exported_path = model.export(format='onnx')
                    
                    results[dataset] = {
                        'status': 'success',
                        'message': 'Model exported',
                        'exported_path': exported_path
                    }
                    logging.info(f"✓ {dataset} model exported: {exported_path}")
                else:
                    results[dataset] = {'status': 'failed', 'message': 'Model not found'}
                    logging.error(f"✗ {dataset} model not found for export")
                
            except Exception as e:
                results[dataset] = {'status': 'error', 'message': str(e)}
                logging.error(f"Error exporting {dataset} model: {e}")
        
        return results
    
    def test_integration(self, datasets: list) -> dict:
        """Test integration with surveillance system"""
        results = {}
        
        for dataset in datasets:
            logging.info(f"Testing {dataset} model integration...")
            
            try:
                # Test object detector with trained model
                model_path = os.path.join(self.output_dir, f'yolov8n_{dataset}_custom', 'weights', 'best.pt')
                
                if os.path.exists(model_path):
                    # Import and test
                    from object_detector import ObjectDetector
                    
                    detector = ObjectDetector(model_path=model_path)
                    
                    # Test with dummy image
                    import numpy as np
                    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    
                    detections = detector.detect_objects(test_image)
                    model_info = detector.get_model_info()
                    
                    results[dataset] = {
                        'status': 'success',
                        'message': 'Integration test passed',
                        'detections': len(detections),
                        'model_info': model_info
                    }
                    logging.info(f"✓ {dataset} integration test passed")
                else:
                    results[dataset] = {'status': 'failed', 'message': 'Model not found for testing'}
                    logging.error(f"✗ {dataset} model not found for integration test")
                
            except Exception as e:
                results[dataset] = {'status': 'error', 'message': str(e)}
                logging.error(f"Error testing {dataset} integration: {e}")
        
        return results
    
    def print_pipeline_summary(self, results: dict):
        """Print pipeline execution summary"""
        logging.info("\n" + "=" * 60)
        logging.info("Pipeline Execution Summary")
        logging.info("=" * 60)
        
        for step, step_results in results.items():
            if step == 'error':
                logging.error(f"Pipeline failed: {step_results}")
                continue
                
            logging.info(f"\n{step.upper()}:")
            
            if isinstance(step_results, dict):
                for dataset, result in step_results.items():
                    status = result.get('status', 'unknown')
                    message = result.get('message', 'No message')
                    
                    if status == 'success':
                        logging.info(f"  ✓ {dataset}: {message}")
                    elif status == 'failed':
                        logging.warning(f"  ✗ {dataset}: {message}")
                    else:
                        logging.error(f"  ? {dataset}: {message}")
        
        # Check overall success
        success_count = 0
        total_count = 0
        
        for step, step_results in results.items():
            if step != 'error' and isinstance(step_results, dict):
                for dataset, result in step_results.items():
                    total_count += 1
                    if result.get('status') == 'success':
                        success_count += 1
        
        if total_count > 0:
            success_rate = (success_count / total_count) * 100
            logging.info(f"\nOverall Success Rate: {success_rate:.1f}% ({success_count}/{total_count})")
            
            if success_rate >= 80:
                logging.info("🎉 Pipeline completed successfully!")
            elif success_rate >= 60:
                logging.info("⚠️  Pipeline completed with some issues")
            else:
                logging.error("❌ Pipeline failed with many issues")
    
    def create_deployment_script(self):
        """Create deployment script for trained models"""
        script_content = '''#!/usr/bin/env python3
"""
Deployment script for trained YOLOv8 models
"""

import os
import shutil
from pathlib import Path

def deploy_models():
    """Deploy trained models to production locations"""
    
    # Source directories
    models_dir = "models"
    
    # Production directories
    production_dir = "production_models"
    os.makedirs(production_dir, exist_ok=True)
    
    # Find trained models
    trained_models = []
    for item in os.listdir(models_dir):
        if item.startswith('yolov8n_') and item.endswith('_custom'):
            model_path = os.path.join(models_dir, item, 'weights', 'best.pt')
            if os.path.exists(model_path):
                trained_models.append((item, model_path))
    
    print(f"Found {len(trained_models)} trained models")
    
    # Deploy each model
    for model_name, model_path in trained_models:
        print(f"Deploying {model_name}...")
        
        # Copy model to production
        dest_path = os.path.join(production_dir, f"{model_name}.pt")
        shutil.copy2(model_path, dest_path)
        
        print(f"  ✓ Deployed to {dest_path}")
    
    # Create model configuration
    config_content = f"""# Model Configuration
available_models = {{
    'avenue': 'production_models/yolov8n_avenue_custom.pt',
    'ucsd': 'production_models/yolov8n_ucsd_custom.pt'
}}

# Usage in surveillance system:
# from object_detector import ObjectDetector
# detector = ObjectDetector(model_path='production_models/yolov8n_avenue_custom.pt')
"""
    
    config_path = os.path.join(production_dir, 'model_config.py')
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✓ Model configuration created: {config_path}")
    print("\\nModels deployed successfully!")

if __name__ == '__main__':
    deploy_models()
'''
        
        script_path = 'deploy_models.py'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        logging.info(f"Deployment script created: {script_path}")
        return script_path

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Complete Training Pipeline for AI Surveillance System')
    parser.add_argument('--datasets', '-d', nargs='+', choices=['avenue', 'ucsd'], default=['avenue', 'ucsd'],
                       help='Datasets to process')
    parser.add_argument('--model-size', '-s', choices=['n', 's', 'm', 'l', 'x'], default='n',
                       help='YOLOv8 model size')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='models', help='Output directory for models')
    parser.add_argument('--skip-download', action='store_true', help='Skip dataset download')
    parser.add_argument('--skip-prepare', action='store_true', help='Skip dataset preparation')
    parser.add_argument('--skip-train', action='store_true', help='Skip model training')
    parser.add_argument('--create-deployment', action='store_true', help='Create deployment script')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = TrainingPipeline(args.data_dir, args.output_dir)
    
    # Run pipeline
    results = pipeline.run_complete_pipeline(args.datasets, args.model_size)
    
    # Create deployment script if requested
    if args.create_deployment:
        pipeline.create_deployment_script()
    
    # Exit with appropriate code
    if 'error' in results:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()
