#!/usr/bin/env python3
"""
Demo script for AI Surveillance System
Demonstrates key features and capabilities
"""

import cv2
import numpy as np
import time
import os
from object_detector import ObjectDetector
from anomaly_detector import AnomalyDetector
import logging

def create_demo_video(output_path: str = "demo_video.mp4", duration: int = 10):
    """Create a demo video with various scenarios"""
    print("Creating demo video...")
    
    # Video parameters
    fps = 30
    width, height = 640, 480
    total_frames = duration * fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create demo scenarios
    scenarios = [
        {'type': 'normal_walking', 'frames': total_frames // 4},
        {'type': 'loitering', 'frames': total_frames // 4},
        {'type': 'unusual_movement', 'frames': total_frames // 4},
        {'type': 'object_abandonment', 'frames': total_frames // 4}
    ]
    
    frame_count = 0
    
    for scenario in scenarios:
        scenario_frames = scenario['frames']
        scenario_type = scenario['type']
        
        print(f"  Generating {scenario_type} scenario...")
        
        for i in range(scenario_frames):
            # Create base frame
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Apply scenario-specific modifications
            if scenario_type == 'normal_walking':
                # Moving person
                x = 100 + int((i / scenario_frames) * 400)
                cv2.rectangle(frame, (x, 200), (x+40, 300), (255, 255, 255), -1)
                cv2.circle(frame, (x+20, 170), 20, (255, 255, 255), -1)
                
            elif scenario_type == 'loitering':
                # Stationary person
                cv2.rectangle(frame, (300, 200), (340, 300), (255, 255, 255), -1)
                cv2.circle(frame, (320, 170), 20, (255, 255, 255), -1)
                
            elif scenario_type == 'unusual_movement':
                # Erratic movement
                x = 200 + int(100 * np.sin(i * 0.2))
                y = 200 + int(50 * np.cos(i * 0.3))
                cv2.rectangle(frame, (x, y), (x+40, y+100), (255, 255, 255), -1)
                cv2.circle(frame, (x+20, y-30), 20, (255, 255, 255), -1)
                
            elif scenario_type == 'object_abandonment':
                # Object appears and stays
                if i > scenario_frames // 2:
                    cv2.rectangle(frame, (400, 250), (450, 300), (0, 255, 0), -1)
            
            # Add timestamp
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Scenario: {scenario_type}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            video_writer.write(frame)
            frame_count += 1
    
    video_writer.release()
    print(f"Demo video created: {output_path}")
    return output_path

def run_demo_detection(video_path: str):
    """Run object detection demo on the video"""
    print("\nRunning object detection demo...")
    
    # Initialize detector
    detector = ObjectDetector()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_count = 0
    total_detections = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection every 10th frame for performance
        if frame_count % 10 == 0:
            detections = detector.detect_objects(frame)
            total_detections += len(detections)
            
            # Draw detections
            annotated_frame = detector.draw_detections(frame, detections)
            
            # Display frame
            cv2.imshow('Object Detection Demo', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Detection demo completed: {total_detections} detections in {frame_count} frames")

def run_demo_anomaly_detection(video_path: str):
    """Run anomaly detection demo on the video"""
    print("\nRunning anomaly detection demo...")
    
    # Initialize components
    detector = ObjectDetector()
    anomaly_detector = AnomalyDetector()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_count = 0
    total_anomalies = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection and anomaly analysis
        detections = detector.detect_objects(frame)
        anomalies = anomaly_detector.analyze_frame(frame, detections, frame_count)
        
        if anomalies:
            total_anomalies += len(anomalies)
            print(f"Frame {frame_count}: {len(anomalies)} anomalies detected")
        
        # Draw detections and anomalies
        annotated_frame = detector.draw_detections(frame, detections)
        
        # Draw anomaly indicators
        for anomaly in anomalies:
            bbox = anomaly.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                color = (0, 0, 255)  # Red for anomalies
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(annotated_frame, anomaly['type'], (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display frame
        cv2.imshow('Anomaly Detection Demo', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Anomaly detection demo completed: {total_anomalies} anomalies detected")

def run_performance_demo():
    """Run performance benchmarking demo"""
    print("\nRunning performance demo...")
    
    # Create test images of different sizes
    test_sizes = [(320, 240), (640, 480), (1280, 720), (1920, 1080)]
    
    detector = ObjectDetector()
    
    for width, height in test_sizes:
        print(f"  Testing {width}x{height} resolution...")
        
        # Create test image
        test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Measure detection time
        start_time = time.time()
        detections = detector.detect_objects(test_image)
        processing_time = time.time() - start_time
        
        fps = 1.0 / processing_time if processing_time > 0 else 0
        
        print(f"    Processing time: {processing_time:.3f}s")
        print(f"    FPS: {fps:.1f}")
        print(f"    Detections: {len(detections)}")

def main():
    """Main demo function"""
    print("=" * 60)
    print("AI-Powered Surveillance System - Demo")
    print("=" * 60)
    
    # Create demo video
    demo_video = create_demo_video()
    
    # Run demos
    try:
        # Object detection demo
        run_demo_detection(demo_video)
        
        # Anomaly detection demo
        run_demo_anomaly_detection(demo_video)
        
        # Performance demo
        run_performance_demo()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        logging.error(f"Demo error: {e}")
    
    # Cleanup
    if os.path.exists(demo_video):
        os.remove(demo_video)
        print(f"Cleaned up demo video: {demo_video}")

if __name__ == '__main__':
    main()
