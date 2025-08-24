#!/usr/bin/env python3
"""
Test Synthetic Video Generation
Quick test to verify the synthetic video generator works
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from synthetic_video_generator import SyntheticVideoGenerator
    print("✅ SyntheticVideoGenerator imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_synthetic_generation():
    """Test basic synthetic video generation"""
    print("\n🚀 Testing Synthetic Video Generation...")
    
    # Initialize generator
    generator = SyntheticVideoGenerator()
    print(f"✅ Generator initialized. Output directory: {generator.output_dir}")
    
    # Test single scenario generation
    print("\n📹 Testing loitering scenario generation...")
    try:
        loitering_path = generator.generate_loitering_scenario()
        if os.path.exists(loitering_path):
            file_size = os.path.getsize(loitering_path) / (1024 * 1024)
            print(f"✅ Loitering scenario generated: {loitering_path}")
            print(f"   File size: {file_size:.1f} MB")
        else:
            print("❌ Loitering scenario file not found")
            return False
    except Exception as e:
        print(f"❌ Loitering scenario generation failed: {e}")
        return False
    
    # Test listing generated videos
    print("\n📋 Testing video listing...")
    try:
        videos = generator.list_generated_videos()
        if videos:
            print(f"✅ Found {len(videos)} generated videos:")
            for video in videos:
                print(f"   - {video['scenario']}: {video['name']} ({video['size_mb']:.1f} MB)")
        else:
            print("❌ No videos found in listing")
            return False
    except Exception as e:
        print(f"❌ Video listing failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Synthetic video generation is working correctly.")
    return True

def test_quick_generation():
    """Test quick generation of all scenarios"""
    print("\n🚀 Testing quick generation of all scenarios...")
    
    generator = SyntheticVideoGenerator()
    
    try:
        results = generator.generate_all_scenarios()
        if results:
            print(f"✅ Generated {len(results)} scenarios:")
            for scenario, path in results.items():
                if os.path.exists(path):
                    file_size = os.path.getsize(path) / (1024 * 1024)
                    print(f"   - {scenario}: {file_size:.1f} MB")
                else:
                    print(f"   - {scenario}: FILE NOT FOUND")
            
            # Check info file
            info_file = generator.output_dir / "scenarios_info.json"
            if info_file.exists():
                print(f"✅ Scenarios info saved to: {info_file}")
            else:
                print("❌ Scenarios info file not found")
                
            return True
        else:
            print("❌ Failed to generate scenarios")
            return False
    except Exception as e:
        print(f"❌ Quick generation failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 SYNTHETIC VIDEO GENERATOR TEST")
    print("=" * 50)
    
    # Test basic functionality
    if test_synthetic_generation():
        print("\n" + "=" * 50)
        print("🎯 BASIC TESTS PASSED - Testing quick generation...")
        print("=" * 50)
        
        # Test quick generation
        if test_quick_generation():
            print("\n🎉 ALL TESTS PASSED! Your synthetic video generator is ready for the hackathon!")
            print("\n📁 Generated videos are in the 'synthetic_videos' directory")
            print("🌐 You can now use the dashboard to generate videos and test your AI system!")
        else:
            print("\n❌ Quick generation tests failed")
    else:
        print("\n❌ Basic tests failed")
        sys.exit(1)
