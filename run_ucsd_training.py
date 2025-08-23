#!/usr/bin/env python3
"""
Run UCSD Training
Simple script to run UCSD training pipeline
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Main function to run UCSD training"""
    print("UCSD Anomaly Detection Dataset Training")
    print("=" * 50)
    
    # Check if dataset exists
    dataset_path = Path("UCSD_Anomaly_Dataset.v1p2")
    if not dataset_path.exists():
        print("❌ UCSD dataset not found!")
        print("Please ensure the dataset is in the current directory.")
        sys.exit(1)
    
    print("✅ UCSD dataset found")
    
    # Check if required files exist
    required_files = [
        "ucsd_dataset_handler.py",
        "ucsd_training_pipeline.py"
    ]
    
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Required file not found: {file}")
            sys.exit(1)
    
    print("✅ All required files found")
    
    # Show options
    print("\nTraining Options:")
    print("1. Quick Test (10 epochs) - Fast validation")
    print("2. Full Training (100 epochs) - Best performance")
    print("3. Fine Tuning (50 epochs) - Balance of speed/quality")
    print("4. Test dataset handler only")
    
    try:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            print("\n🚀 Starting Quick Test Training...")
            cmd = [sys.executable, "ucsd_training_pipeline.py", "--mode", "quick_test", "--model-size", "n"]
            
        elif choice == "2":
            print("\n🚀 Starting Full Training...")
            cmd = [sys.executable, "ucsd_training_pipeline.py", "--mode", "full_training", "--model-size", "n"]
            
        elif choice == "3":
            print("\n🚀 Starting Fine Tuning...")
            cmd = [sys.executable, "ucsd_training_pipeline.py", "--mode", "fine_tuning", "--model-size", "n"]
            
        elif choice == "4":
            print("\n🧪 Testing Dataset Handler...")
            cmd = [sys.executable, "test_ucsd_dataset.py"]
            
        else:
            print("❌ Invalid choice. Please select 1-4.")
            return
        
        # Run the command
        print(f"Running: {' '.join(cmd)}")
        print("-" * 50)
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Training completed successfully!")
        else:
            print(f"\n❌ Training failed with return code: {result.returncode}")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
