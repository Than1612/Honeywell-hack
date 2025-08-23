import cv2
import numpy as np
import time
import logging
import threading
import requests
import json
from typing import Optional, Dict, Any, List
import argparse
import os
from datetime import datetime

from object_detector import ObjectDetector
from anomaly_detector import AnomalyDetector

class SurveillanceSystem:
    """
    Main AI-powered surveillance system
    Integrates object detection, anomaly detection, and dashboard communication
    """
    
    def __init__(self, 
                 video_source: str = "0",
                 dashboard_url: str = "http://localhost:5000",
                 save_video: bool = True,
                 output_dir: str = "output"):
        """
        Initialize surveillance system
        
        Args:
            video_source: Video source (0 for webcam, file path for video file)
            dashboard_url: URL of the dashboard
            save_video: Whether to save processed video
            output_dir: Directory to save output files
        """
        self.video_source = video_source
        self.dashboard_url = dashboard_url
        self.save_video = save_video
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.object_detector = ObjectDetector()
        self.anomaly_detector = AnomalyDetector()
        
        # Video processing variables
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        
        # Video writer for saving
        self.video_writer = None
        self.video_fps = 30
        self.video_size = None
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_anomalies': 0,
            'fps': 0,
            'active_person_tracks': 0,
            'active_object_tracks': 0
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'surveillance.log')),
                logging.StreamHandler()
            ]
        )
        
        logging.info("Surveillance system initialized")
    
    def start(self):
        """Start the surveillance system"""
        try:
            # Initialize video capture
            self._initialize_video()
            
            if self.cap is None or not self.cap.isOpened():
                logging.error("Failed to open video source")
                return
            
            # Initialize video writer if saving
            if self.save_video:
                self._initialize_video_writer()
            
            logging.info("Starting surveillance system...")
            
            # Start dashboard update thread
            dashboard_thread = threading.Thread(target=self._dashboard_update_loop, daemon=True)
            dashboard_thread.start()
            
            # Main processing loop
            self._process_video()
            
        except KeyboardInterrupt:
            logging.info("Surveillance system stopped by user")
        except Exception as e:
            logging.error(f"Error in surveillance system: {e}")
        finally:
            self._cleanup()
    
    def _initialize_video(self):
        """Initialize video capture"""
        try:
            if self.video_source.isdigit():
                # Webcam
                self.cap = cv2.VideoCapture(int(self.video_source))
                logging.info(f"Initialized webcam {self.video_source}")
            else:
                # Video file
                self.cap = cv2.VideoCapture(self.video_source)
                logging.info(f"Initialized video file: {self.video_source}")
            
            if not self.cap.isOpened():
                raise Exception("Failed to open video source")
            
            # Get video properties
            self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            if self.video_fps == 0:
                self.video_fps = 30  # Default FPS
            
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_size = (width, height)
            
            logging.info(f"Video properties: {width}x{height} @ {self.video_fps} FPS")
            
        except Exception as e:
            logging.error(f"Error initializing video: {e}")
            raise
    
    def _initialize_video_writer(self):
        """Initialize video writer for saving processed video"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"surveillance_{timestamp}.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, self.video_fps, self.video_size
            )
            
            logging.info(f"Video writer initialized: {output_path}")
            
        except Exception as e:
            logging.error(f"Error initializing video writer: {e}")
            self.save_video = False
    
    def _process_video(self):
        """Main video processing loop"""
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logging.info("End of video stream")
                break
            
            frame_count += 1
            
            # Process frame
            processed_frame, detections, anomalies = self._process_frame(frame, frame_count)
            
            # Update statistics
            self._update_stats(detections, anomalies)
            
            # Save frame if enabled
            if self.save_video and self.video_writer is not None:
                self.video_writer.write(processed_frame)
            
            # Display frame (optional)
            cv2.imshow('AI Surveillance System', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_screenshot(processed_frame, frame_count)
            
            # Calculate FPS
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                self.fps = frame_count / elapsed_time
                self.stats['fps'] = self.fps
        
        logging.info("Video processing completed")
    
    def _process_frame(self, frame: np.ndarray, frame_number: int) -> tuple:
        """
        Process a single frame
        
        Returns:
            tuple: (processed_frame, detections, anomalies)
        """
        # Object detection
        detections = self.object_detector.detect_objects(frame)
        
        # Anomaly detection
        anomalies = self.anomaly_detector.analyze_frame(frame, detections, frame_number)
        
        # Draw detections and anomalies on frame
        processed_frame = self._draw_frame_annotations(frame, detections, anomalies)
        
        return processed_frame, detections, anomalies
    
    def _draw_frame_annotations(self, frame: np.ndarray, detections: List[Dict], anomalies: List[Dict]) -> np.ndarray:
        """Draw detections and anomalies on frame"""
        # Draw object detections
        frame = self.object_detector.draw_detections(frame, detections)
        
        # Draw anomaly indicators
        for anomaly in anomalies:
            bbox = anomaly.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                
                # Draw anomaly bounding box
                color = self._get_anomaly_color(anomaly['type'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw anomaly label
                label = f"{anomaly['type'].upper()}: {anomaly['severity']}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw timestamp
                timestamp = datetime.now().strftime("%H:%M:%S")
                cv2.putText(frame, timestamp, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
    def _get_anomaly_color(self, anomaly_type: str) -> tuple:
        """Get color for different anomaly types"""
        colors = {
            'loitering': (0, 255, 255),      # Yellow
            'unusual_movement': (0, 0, 255),  # Red
            'object_abandonment': (255, 0, 255),  # Magenta
            'erratic_movement': (0, 165, 255)  # Orange
        }
        return colors.get(anomaly_type, (0, 255, 0))  # Default green
    
    def _update_stats(self, detections: List[Dict], anomalies: List[Dict]):
        """Update system statistics"""
        self.stats['total_frames'] += 1
        self.stats['total_detections'] += len(detections)
        self.stats['total_anomalies'] += len(anomalies)
        
        # Get tracking stats
        tracking_stats = self.anomaly_detector.get_tracking_stats()
        self.stats['active_person_tracks'] = tracking_stats.get('active_person_tracks', 0)
        self.stats['active_object_tracks'] = tracking_stats.get('active_object_tracks', 0)
    
    def _dashboard_update_loop(self):
        """Background thread for updating dashboard"""
        while True:
            try:
                self._update_dashboard()
                time.sleep(1)  # Update every second
            except Exception as e:
                logging.error(f"Error updating dashboard: {e}")
                time.sleep(5)  # Wait longer on error
    
    def _update_dashboard(self):
        """Send data to dashboard"""
        try:
            # Prepare data for dashboard
            dashboard_data = {
                'detections': self.object_detector.get_detection_stats(),
                'anomalies': self.anomaly_detector.get_anomaly_summary(),
                'stats': self.stats
            }
            
            # Send to dashboard
            response = requests.post(
                f"{self.dashboard_url}/api/update",
                json=dashboard_data,
                timeout=5
            )
            
            if response.status_code != 200:
                logging.warning(f"Dashboard update failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logging.debug(f"Dashboard not accessible: {e}")
        except Exception as e:
            logging.error(f"Error updating dashboard: {e}")
    
    def _save_screenshot(self, frame: np.ndarray, frame_number: int):
        """Save current frame as screenshot"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}_frame_{frame_number}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            cv2.imwrite(filepath, frame)
            logging.info(f"Screenshot saved: {filepath}")
            
        except Exception as e:
            logging.error(f"Error saving screenshot: {e}")
    
    def _cleanup(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        
        if self.video_writer is not None:
            self.video_writer.release()
        
        cv2.destroyAllWindows()
        
        logging.info("Surveillance system cleanup completed")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='AI-Powered Surveillance System')
    parser.add_argument('--source', '-s', default='0', 
                       help='Video source (0 for webcam, file path for video)')
    parser.add_argument('--dashboard', '-d', default='http://localhost:5000',
                       help='Dashboard URL')
    parser.add_argument('--no-save', action='store_true',
                       help='Disable video saving')
    parser.add_argument('--output', '-o', default='output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create and start surveillance system
    system = SurveillanceSystem(
        video_source=args.source,
        dashboard_url=args.dashboard,
        save_video=not args.no_save,
        output_dir=args.output
    )
    
    try:
        system.start()
    except KeyboardInterrupt:
        logging.info("System stopped by user")

if __name__ == '__main__':
    main()
