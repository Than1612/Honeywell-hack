#!/usr/bin/env python3
"""
Test UCSD Dataset Handler
Simple script to test the UCSD dataset handler functionality
"""

import os
import sys
from pathlib import Path

def test_ucsd_dataset():
    """Test the UCSD dataset handler"""
    try:
        print("Testing UCSD Dataset Handler...")
        print("="*50)
        
        # Check if dataset exists
        dataset_path = Path("UCSD_Anomaly_Dataset.v1p2")
        if not dataset_path.exists():
            print(f"❌ Dataset not found at: {dataset_path}")
            return False
        
        print(f"✅ Dataset found at: {dataset_path}")
        
        # Check dataset structure
        required_paths = [
            dataset_path / "UCSDped1" / "Train",
            dataset_path / "UCSDped1" / "Test",
            dataset_path / "UCSDped2" / "Train", 
            dataset_path / "UCSDped2" / "Test"
        ]
        
        for path in required_paths:
            if path.exists():
                print(f"✅ {path} exists")
            else:
                print(f"❌ {path} missing")
                return False
        
        # Count files
        print("\nDataset Statistics:")
        print("-" * 30)
        
        for peds_name in ["UCSDped1", "UCSDped2"]:
            peds_path = dataset_path / peds_name
            
            for split in ["Train", "Test"]:
                split_path = peds_path / split
                clip_count = len([d for d in split_path.iterdir() if d.is_dir() and not d.name.endswith("_gt")])
                
                # Count frames in first few clips
                frame_count = 0
                for clip_dir in list(split_path.glob("*"))[:3]:  # Check first 3 clips
                    if clip_dir.is_dir() and not clip_dir.name.endswith("_gt"):
                        tif_files = list(clip_dir.glob("*.tif"))
                        frame_count += len(tif_files)
                
                print(f"{peds_name}/{split}: {clip_count} clips, ~{frame_count} frames (sample)")
        
        # Test dataset handler import
        print("\nTesting Dataset Handler Import...")
        print("-" * 30)
        
        try:
            from ucsd_dataset_handler import UCSDDatasetHandler
            print("✅ UCSD Dataset Handler imported successfully")
            
            # Initialize handler
            handler = UCSDDatasetHandler()
            print("✅ Handler initialized successfully")
            
            # Get dataset info
            info = handler.get_dataset_info()
            print("✅ Dataset info retrieved successfully")
            print(f"   Total frames: {info['total_frames']}")
            
            # Test quick dataset creation
            print("\nTesting Quick Dataset Creation...")
            print("-" * 30)
            
            quick_dataset_path = handler.create_quick_test_dataset()
            print(f"✅ Quick test dataset created at: {quick_dataset_path}")
            
            # Check if files were created
            quick_path = Path(quick_dataset_path)
            if quick_path.exists():
                train_images = list((quick_path / "images" / "train").glob("*.jpg"))
                train_labels = list((quick_path / "labels" / "train").glob("*.txt"))
                val_images = list((quick_path / "images" / "val").glob("*.jpg"))
                val_labels = list((quick_path / "labels" / "val").glob("*.txt"))
                
                print(f"   Training: {len(train_images)} images, {len(train_labels)} labels")
                print(f"   Validation: {len(val_images)} images, {len(val_labels)} labels")
                
                # Check dataset.yaml
                dataset_yaml = quick_path / "dataset.yaml"
                if dataset_yaml.exists():
                    print(f"   Dataset YAML: ✅")
                else:
                    print(f"   Dataset YAML: ❌")
            
            print("\n🎉 All tests passed! UCSD dataset is ready for training.")
            return True
            
        except ImportError as e:
            print(f"❌ Failed to import UCSD Dataset Handler: {e}")
            return False
        except Exception as e:
            print(f"❌ Error testing dataset handler: {e}")
            return False
    
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def main():
    """Main function"""
    success = test_ucsd_dataset()
    
    if success:
        print("\n" + "="*50)
        print("UCSD DATASET READY FOR TRAINING!")
        print("="*50)
        print("Next steps:")
        print("1. Run: python ucsd_training_pipeline.py --mode quick_test")
        print("2. Check the generated models in the 'models/' directory")
        print("3. Use the trained model in your surveillance system")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("UCSD DATASET TEST FAILED!")
        print("="*50)
        print("Please check:")
        print("1. Dataset path and structure")
        print("2. Required dependencies")
        print("3. File permissions")
        print("="*50)
        sys.exit(1)

if __name__ == "__main__":
    main()
