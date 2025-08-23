#!/usr/bin/env python3
"""
UCSD Anomaly Dataset Handler
Specialized handler for the UCSD Anomaly Detection Dataset
Converts .tif files to YOLO format for YOLOv8 training
"""

import os
import cv2
import numpy as np
import shutil
from pathlib import Path
import logging
from typing import List, Tuple, Dict
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UCSDDatasetHandler:
    """Handler for UCSD Anomaly Detection Dataset"""
    
    def __init__(self, dataset_root: str = "UCSD_Anomaly_Dataset.v1p2"):
        self.dataset_root = Path(dataset_root)
        self.peds1_path = self.dataset_root / "UCSDped1"
        self.peds2_path = self.dataset_root / "UCSDped2"
        
        # Validate dataset structure
        if not self._validate_dataset():
            raise ValueError("Invalid UCSD dataset structure")
        
        logger.info(f"UCSD Dataset initialized from: {self.dataset_root}")
    
    def _validate_dataset(self) -> bool:
        """Validate the dataset structure"""
        required_paths = [
            self.peds1_path / "Train",
            self.peds1_path / "Test", 
            self.peds2_path / "Train",
            self.peds2_path / "Test"
        ]
        
        for path in required_paths:
            if not path.exists():
                logger.error(f"Required path not found: {path}")
                return False
        
        logger.info("Dataset structure validated successfully")
        return True
    
    def get_dataset_info(self) -> Dict:
        """Get information about the dataset"""
        info = {
            'peds1': {
                'train_clips': len(list((self.peds1_path / "Train").glob("Train*"))),
                'test_clips': len(list((self.peds1_path / "Test").glob("Test*"))),
                'test_with_gt': len(list((self.peds1_path / "Test").glob("*_gt")))
            },
            'peds2': {
                'train_clips': len(list((self.peds2_path / "Train").glob("Train*"))),
                'test_clips': len(list((self.peds2_path / "Test").glob("Test*")))
            }
        }
        
        # Count total frames
        total_frames = 0
        for peds_path in [self.peds1_path, self.peds2_path]:
            for split in ["Train", "Test"]:
                split_path = peds_path / split
                for clip_dir in split_path.glob("*"):
                    if clip_dir.is_dir() and not clip_dir.name.endswith("_gt"):
                        total_frames += len(list(clip_dir.glob("*.tif")))
        
        info['total_frames'] = total_frames
        return info
    
    def prepare_yolo_dataset(self, output_dir: str = "processed_data/ucsd_yolo", 
                           train_split: float = 0.8) -> str:
        """
        Prepare UCSD dataset for YOLO training
        
        Args:
            output_dir: Output directory for YOLO format data
            train_split: Fraction of data to use for training
            
        Returns:
            Path to the created dataset.yaml file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create YOLO directory structure
        yolo_dirs = {
            'train': output_path / 'images' / 'train',
            'val': output_path / 'images' / 'val',
            'train_labels': output_path / 'labels' / 'train',
            'val_labels': output_path / 'labels' / 'val'
        }
        
        for dir_path in yolo_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Process Peds1 and Peds2
        all_clips = []
        
        # Collect all training clips
        for peds_path in [self.peds1_path, self.peds2_path]:
            train_path = peds_path / "Train"
            for clip_dir in train_path.glob("*"):
                if clip_dir.is_dir():
                    all_clips.append(('train', peds_path.name, clip_dir))
        
        # Collect all testing clips
        for peds_path in [self.peds1_path, self.peds2_path]:
            test_path = peds_path / "Test"
            for clip_dir in test_path.glob("*"):
                if clip_dir.is_dir() and not clip_dir.name.endswith("_gt"):
                    all_clips.append(('test', peds_path.name, clip_dir))
        
        # Shuffle and split
        np.random.shuffle(all_clips)
        split_idx = int(len(all_clips) * train_split)
        train_clips = all_clips[:split_idx]
        val_clips = all_clips[split_idx:]
        
        logger.info(f"Processing {len(train_clips)} training clips and {len(val_clips)} validation clips")
        
        # Process training data
        self._process_clips(train_clips, yolo_dirs['train'], yolo_dirs['train_labels'])
        
        # Process validation data  
        self._process_clips(val_clips, yolo_dirs['val'], yolo_dirs['val_labels'])
        
        # Create dataset.yaml
        dataset_yaml = self._create_dataset_yaml(output_path)
        
        logger.info(f"YOLO dataset created successfully at: {output_path}")
        return str(dataset_yaml)
    
    def _process_clips(self, clips: List[Tuple], images_dir: Path, labels_dir: Path):
        """Process clips and convert to YOLO format"""
        frame_count = 0
        
        for split, peds_name, clip_dir in clips:
            logger.info(f"Processing {peds_name}/{clip_dir.name}")
            
            # Get all .tif files in the clip
            tif_files = sorted(clip_dir.glob("*.tif"))
            
            for tif_file in tif_files:
                # Convert .tif to .jpg for YOLO
                img_path = images_dir / f"{peds_name}_{clip_dir.name}_{tif_file.stem}.jpg"
                self._convert_tif_to_jpg(tif_file, img_path)
                
                # Create YOLO label (person detection in center)
                label_path = labels_dir / f"{peds_name}_{clip_dir.name}_{tif_file.stem}.txt"
                self._create_yolo_label(label_path, img_path)
                
                frame_count += 1
        
        logger.info(f"Processed {frame_count} frames to {images_dir}")
    
    def _convert_tif_to_jpg(self, tif_path: Path, jpg_path: Path):
        """Convert .tif file to .jpg"""
        try:
            # Read .tif file
            img = cv2.imread(str(tif_path), cv2.IMREAD_COLOR)
            if img is None:
                # Try reading as grayscale and convert to RGB
                img = cv2.imread(str(tif_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    logger.warning(f"Could not read {tif_path}")
                    return
            
            # Save as .jpg
            cv2.imwrite(str(jpg_path), img)
            
        except Exception as e:
            logger.error(f"Error converting {tif_path}: {e}")
    
    def _create_yolo_label(self, label_path: Path, img_path: Path):
        """Create YOLO format label for person detection"""
        try:
            # Read image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                return
            
            height, width = img.shape[:2]
            
            # Create synthetic label: person in center of frame
            # YOLO format: <class> <x_center> <y_center> <width> <height>
            # Class 0 = person
            
            # Place person in center with reasonable size
            x_center = 0.5  # Center of frame
            y_center = 0.5  # Center of frame
            w = 0.3         # 30% of frame width
            h = 0.6         # 60% of frame height
            
            # Write label file
            with open(label_path, 'w') as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                
        except Exception as e:
            logger.error(f"Error creating label {label_path}: {e}")
    
    def _create_dataset_yaml(self, output_path: Path) -> Path:
        """Create dataset.yaml file for YOLOv8 training"""
        yaml_content = f"""# UCSD Anomaly Detection Dataset - YOLO Format
# Generated by UCSDDatasetHandler

path: {output_path.absolute()}  # Dataset root directory
train: images/train  # Train images (relative to 'path')
val: images/val      # Val images (relative to 'path')

# Classes
names:
  0: person

# Dataset information
nc: 1  # Number of classes
"""
        
        yaml_path = output_path / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        logger.info(f"Created dataset.yaml at: {yaml_path}")
        return yaml_path
    
    def get_training_samples(self, num_samples: int = 10) -> List[Tuple[str, str]]:
        """Get sample training data for testing"""
        samples = []
        
        for peds_path in [self.peds1_path, self.peds2_path]:
            train_path = peds_path / "Train"
            for clip_dir in train_path.glob("*"):
                if clip_dir.is_dir():
                    tif_files = list(clip_dir.glob("*.tif"))[:5]  # Take first 5 frames
                    for tif_file in tif_files:
                        samples.append((str(tif_file), f"{peds_path.name}/{clip_dir.name}"))
                        if len(samples) >= num_samples:
                            return samples
        
        return samples
    
    def create_quick_test_dataset(self, output_dir: str = "processed_data/ucsd_quick_test") -> str:
        """Create a small test dataset for quick validation"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create minimal structure
        yolo_dirs = {
            'train': output_path / 'images' / 'train',
            'val': output_path / 'images' / 'val',
            'train_labels': output_path / 'labels' / 'train',
            'val_labels': output_path / 'labels' / 'val'
        }
        
        for dir_path in yolo_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Process just a few clips for quick testing
        quick_clips = []
        
        # Take first 2 training clips from each dataset
        for peds_path in [self.peds1_path, self.peds2_path]:
            train_path = peds_path / "Train"
            for clip_dir in list(train_path.glob("*"))[:2]:
                if clip_dir.is_dir():
                    quick_clips.append(('train', peds_path.name, clip_dir))
        
        # Take first 2 test clips from each dataset
        for peds_path in [self.peds1_path, self.peds2_path]:
            test_path = peds_path / "Test"
            for clip_dir in list(test_path.glob("*"))[:2]:
                if clip_dir.is_dir() and not clip_dir.name.endswith("_gt"):
                    quick_clips.append(('test', peds_path.name, clip_dir))
        
        # Split for quick test
        split_idx = len(quick_clips) // 2
        train_clips = quick_clips[:split_idx]
        val_clips = quick_clips[split_idx:]
        
        logger.info(f"Creating quick test dataset with {len(train_clips)} train and {len(val_clips)} val clips")
        
        # Process clips
        self._process_clips(train_clips, yolo_dirs['train'], yolo_dirs['train_labels'])
        self._process_clips(val_clips, yolo_dirs['val'], yolo_dirs['val_labels'])
        
        # Create dataset.yaml
        dataset_yaml = self._create_dataset_yaml(output_path)
        
        logger.info(f"Quick test dataset created at: {output_path}")
        return str(dataset_yaml)


def main():
    """Main function for testing the dataset handler"""
    try:
        # Initialize handler
        handler = UCSDDatasetHandler()
        
        # Get dataset info
        info = handler.get_dataset_info()
        print("Dataset Information:")
        print(json.dumps(info, indent=2))
        
        # Create quick test dataset
        print("\nCreating quick test dataset...")
        test_dataset_path = handler.create_quick_test_dataset()
        print(f"Quick test dataset created at: {test_dataset_path}")
        
        # Create full dataset
        print("\nCreating full YOLO dataset...")
        full_dataset_path = handler.prepare_yolo_dataset()
        print(f"Full dataset created at: {full_dataset_path}")
        
        print("\nDataset preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
