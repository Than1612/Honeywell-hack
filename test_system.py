#!/usr/bin/env python3
"""
Test script for AI Surveillance System
"""

import cv2
import numpy as np
import time
import logging
from object_detector import ObjectDetector
from anomaly_detector import AnomalyDetector

def test_object_detection():
    """Test object detection functionality"""
    print("Testing object detection...")
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Add simple objects
    cv2.rectangle(test_image, (200, 150), (280, 350), (255, 255, 255), -1)
    cv2.circle(test_image, (240, 120), 30, (255, 255, 255), -1)
    
    # Initialize detector
    detector = ObjectDetector()
    
    # Test detection
    start_time = time.time()
    detections = detector.detect_objects(test_image)
    processing_time = time.time() - start_time
    
    print(f"  Processing time: {processing_time:.3f}s")
    print(f"  Detections found: {len(detections)}")
    
    # Test drawing
    annotated_image = detector.draw_detections(test_image, detections)
    
    return len(detections) > 0

def test_anomaly_detection():
    """Test anomaly detection functionality"""
    print("Testing anomaly detection...")
    
    # Create test sequence
    test_frames = []
    for i in range(30):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add stationary person (loitering scenario)
        cv2.rectangle(frame, (300, 200), (340, 300), (255, 255, 255), -1)
        cv2.circle(frame, (320, 170), 20, (255, 255, 255), -1)
        
        test_frames.append(frame)
    
    # Initialize detector
    detector = ObjectDetector()
    anomaly_detector = AnomalyDetector()
    
    anomalies_found = 0
    
    for i, frame in enumerate(test_frames):
        detections = detector.detect_objects(frame)
        anomalies = anomaly_detector.analyze_frame(frame, detections, i)
        anomalies_found += len(anomalies)
    
    print(f"  Total anomalies detected: {anomalies_found}")
    
    return anomalies_found > 0

def test_integration():
    """Test system integration"""
    print("Testing system integration...")
    
    try:
        # Test imports
        from main import SurveillanceSystem
        print("  ✓ Main system imported successfully")
        
        # Test configuration
        from config.config import Config
        print("  ✓ Configuration loaded successfully")
        
        # Test utilities
        from utils.video_utils import resize_frame
        print("  ✓ Utilities loaded successfully")
        
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("AI Surveillance System - System Test")
    print("=" * 50)
    
    tests = [
        ("Object Detection", test_object_detection),
        ("Anomaly Detection", test_anomaly_detection),
        ("System Integration", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                print(f"  ✓ {test_name} test PASSED")
                passed += 1
            else:
                print(f"  ✗ {test_name} test FAILED")
        except Exception as e:
            print(f"  ✗ {test_name} test ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    print("=" * 50)

if __name__ == '__main__':
    main()
