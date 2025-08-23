import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Tuple, Dict, Any
import time
import logging
import os

class ObjectDetector:
    """
    YOLOv8-based object detection for surveillance system
    Uses custom models trained on Avenue and UCSD datasets
    """
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        """
        Initialize the object detector
        
        Args:
            model_path: Path to custom trained YOLOv8 model
            confidence_threshold: Minimum confidence for detection
        """
        self.confidence_threshold = confidence_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model paths for custom trained models
        self.model_paths = {
            'avenue': 'models/yolov8n_avenue_custom/weights/best.pt',
            'ucsd': 'models/yolov8n_ucsd_quick_test/weights/best.pt',
            'ucsd_full': 'models/yolov8n_ucsd_full_training/weights/best.pt',
            'default': 'yolov8n.pt'
        }
        
        # Load YOLOv8 model
        try:
            if model_path and os.path.exists(model_path):
                # Use specified custom model
                self.model = YOLO(model_path)
                logging.info(f"Custom YOLOv8 model loaded from {model_path}")
            else:
                # Try to load custom trained models
                self.model = self._load_best_available_model()
            
            logging.info(f"YOLOv8 model loaded successfully on {self.device}")
        except Exception as e:
            logging.error(f"Failed to load YOLOv8 model: {e}")
            raise
        
        # Person class ID in COCO dataset
        self.person_class_id = 0
        
        # Initialize tracking variables
        self.detection_history = []
        self.max_history_length = 30
        
        # Model performance metrics
        self.inference_times = []
        self.max_inference_history = 100
        
    def _load_best_available_model(self) -> YOLO:
        """Load the best available trained model"""
        # Check for custom trained models
        for dataset_name, model_path in self.model_paths.items():
            if dataset_name != 'default' and os.path.exists(model_path):
                logging.info(f"Loading custom trained model: {dataset_name}")
                return YOLO(model_path)
        
        # Fall back to default model
        logging.info("No custom trained models found, using default YOLOv8n")
        return YOLO(self.model_paths['default'])
    
    def get_available_models(self) -> Dict[str, str]:
        """Get list of available trained models"""
        available = {}
        
        for dataset_name, model_path in self.model_paths.items():
            if dataset_name != 'default' and os.path.exists(model_path):
                available[dataset_name] = model_path
            elif dataset_name == 'default':
                available[dataset_name] = 'Default YOLOv8n (pretrained)'
        
        return available
    
    def switch_model(self, model_path: str) -> bool:
        """Switch to a different model"""
        try:
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                logging.info(f"Switched to model: {model_path}")
                return True
            else:
                logging.error(f"Model not found: {model_path}")
                return False
        except Exception as e:
            logging.error(f"Failed to switch model: {e}")
            return False
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in a frame using YOLOv8
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of detected objects with bounding boxes and confidence scores
        """
        try:
            start_time = time.time()
            
            # Run YOLOv8 inference
            results = self.model(frame, verbose=False)
            
            inference_time = time.time() - start_time
            self._update_inference_time(inference_time)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Filter by confidence threshold
                        if confidence >= self.confidence_threshold:
                            detection = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence),
                                'class_id': class_id,
                                'class_name': self.model.names[class_id],
                                'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                                'inference_time': inference_time
                            }
                            detections.append(detection)
            
            # Update detection history
            self._update_detection_history(detections)
            
            return detections
            
        except Exception as e:
            logging.error(f"Error in object detection: {e}")
            return []
    
    def detect_persons(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect only persons in a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of detected persons
        """
        detections = self.detect_objects(frame)
        return [det for det in detections if det['class_id'] == self.person_class_id]
    
    def get_person_trajectories(self) -> List[List[Tuple[int, int]]]:
        """
        Get trajectories of detected persons over time
        
        Returns:
            List of trajectories (each trajectory is a list of center points)
        """
        if not self.detection_history:
            return []
        
        trajectories = []
        current_trajectories = {}
        
        for frame_detections in self.detection_history:
            # Match detections with existing trajectories using IoU
            for detection in frame_detections:
                if detection['class_id'] == self.person_class_id:
                    matched = False
                    
                    for track_id, trajectory in current_trajectories.items():
                        if self._calculate_iou(detection['bbox'], trajectory[-1]) > 0.3:
                            trajectory.append(detection['center'])
                            matched = True
                            break
                    
                    if not matched:
                        # Start new trajectory
                        track_id = len(current_trajectories)
                        current_trajectories[track_id] = [detection['center']]
        
        # Convert to list format
        for trajectory in current_trajectories.values():
            if len(trajectory) > 1:  # Only keep trajectories with multiple points
                trajectories.append(trajectory)
        
        return trajectories
    
    def _update_detection_history(self, detections: List[Dict[str, Any]]):
        """Update detection history for trajectory analysis"""
        self.detection_history.append(detections)
        
        # Keep only recent history
        if len(self.detection_history) > self.max_history_length:
            self.detection_history.pop(0)
    
    def _update_inference_time(self, inference_time: float):
        """Update inference time history for performance monitoring"""
        self.inference_times.append(inference_time)
        
        # Keep only recent history
        if len(self.inference_times) > self.max_inference_history:
            self.inference_times.pop(0)
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detection bounding boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with drawn detections
        """
        frame_copy = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            color = (0, 255, 0) if detection['class_id'] == self.person_class_id else (255, 0, 0)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame_copy, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            center_x, center_y = detection['center']
            cv2.circle(frame_copy, (center_x, center_y), 3, (0, 0, 255), -1)
        
        return frame_copy
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get statistics about recent detections"""
        if not self.detection_history:
            return {}
        
        total_detections = sum(len(frame_dets) for frame_dets in self.detection_history)
        person_detections = sum(
            len([det for det in frame_dets if det['class_id'] == self.person_class_id])
            for frame_dets in self.detection_history
        )
        
        # Performance metrics
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        return {
            'total_detections': total_detections,
            'person_detections': person_detections,
            'frames_processed': len(self.detection_history),
            'avg_detections_per_frame': total_detections / len(self.detection_history) if self.detection_history else 0,
            'avg_inference_time': avg_inference_time,
            'fps': fps,
            'available_models': self.get_available_models()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        try:
            model_info = {
                'model_path': str(self.model.ckpt_path) if hasattr(self.model, 'ckpt_path') else 'Unknown',
                'model_type': type(self.model).__name__,
                'device': self.device,
                'num_classes': len(self.model.names) if hasattr(self.model, 'names') else 'Unknown',
                'class_names': list(self.model.names.values()) if hasattr(self.model, 'names') else []
            }
            return model_info
        except Exception as e:
            logging.error(f"Error getting model info: {e}")
            return {'error': str(e)}
    
    def benchmark_performance(self, test_images: List[np.ndarray]) -> Dict[str, Any]:
        """Benchmark model performance on test images"""
        if not test_images:
            return {}
        
        inference_times = []
        detection_counts = []
        
        for image in test_images:
            start_time = time.time()
            detections = self.detect_objects(image)
            inference_time = time.time() - start_time
            
            inference_times.append(inference_time)
            detection_counts.append(len(detections))
        
        benchmark_results = {
            'num_images': len(test_images),
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'avg_fps': 1.0 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0,
            'avg_detections': np.mean(detection_counts),
            'total_detections': sum(detection_counts)
        }
        
        return benchmark_results
