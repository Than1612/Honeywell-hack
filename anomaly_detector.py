import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Tuple
import time
import logging
from collections import deque
import math

class AnomalyDetector:
    """
    Behavioral anomaly detection for surveillance system
    Detects: loitering, unusual movements, object abandonment
    """
    
    def __init__(self, 
                 loitering_threshold: int = 50,
                 movement_threshold: float = 0.1,
                 abandonment_threshold: int = 100):
        """
        Initialize anomaly detector
        
        Args:
            loitering_threshold: Frames threshold for loitering detection
            movement_threshold: Movement threshold for unusual behavior
            abandonment_threshold: Frames threshold for object abandonment
        """
        self.loitering_threshold = loitering_threshold
        self.movement_threshold = movement_threshold
        self.abandonment_threshold = abandonment_threshold
        
        # Initialize anomaly detection models
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
        # Tracking variables
        self.person_tracks = {}  # Track each person's behavior
        self.object_tracks = {}  # Track objects for abandonment detection
        self.anomaly_history = []
        self.max_history = 1000
        
        # Movement analysis
        self.movement_patterns = deque(maxlen=100)
        self.velocity_history = deque(maxlen=50)
        
        # Initialize scaler with dummy data
        dummy_data = np.random.randn(100, 5)
        self.scaler.fit(dummy_data)
        
        logging.info("Anomaly detector initialized successfully")
    
    def analyze_frame(self, 
                     frame: np.ndarray, 
                     detections: List[Dict[str, Any]], 
                     frame_number: int) -> List[Dict[str, Any]]:
        """
        Analyze frame for behavioral anomalies
        
        Args:
            frame: Current frame
            detections: Object detections from YOLOv8
            frame_number: Current frame number
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Update person tracks
        self._update_person_tracks(detections, frame_number)
        
        # Update object tracks
        self._update_object_tracks(detections, frame_number)
        
        # Detect loitering
        loitering_anomalies = self._detect_loitering(frame_number)
        anomalies.extend(loitering_anomalies)
        
        # Detect unusual movements
        movement_anomalies = self._detect_unusual_movements(frame_number)
        anomalies.extend(movement_anomalies)
        
        # Detect object abandonment
        abandonment_anomalies = self._detect_object_abandonment(frame_number)
        anomalies.extend(abandonment_anomalies)
        
        # Update anomaly history
        for anomaly in anomalies:
            self._add_anomaly(anomaly)
        
        return anomalies
    
    def _update_person_tracks(self, detections: List[Dict[str, Any]], frame_number: int):
        """Update tracking information for detected persons"""
        current_persons = {}
        
        for detection in detections:
            if detection['class_id'] == 0:  # Person class
                bbox = detection['bbox']
                center = detection['center']
                
                # Find matching track using IoU
                matched_track_id = None
                for track_id, track in self.person_tracks.items():
                    if self._calculate_iou(bbox, track['last_bbox']) > 0.3:
                        matched_track_id = track_id
                        break
                
                if matched_track_id is not None:
                    # Update existing track
                    track = self.person_tracks[matched_track_id]
                    track['frames_present'].append(frame_number)
                    track['positions'].append(center)
                    track['last_bbox'] = bbox
                    track['last_seen'] = frame_number
                    track['movement'] = self._calculate_movement(track['positions'])
                    
                    current_persons[matched_track_id] = track
                else:
                    # Create new track
                    track_id = len(self.person_tracks)
                    self.person_tracks[track_id] = {
                        'frames_present': [frame_number],
                        'positions': [center],
                        'last_bbox': bbox,
                        'last_seen': frame_number,
                        'movement': 0.0,
                        'start_frame': frame_number
                    }
                    current_persons[track_id] = self.person_tracks[track_id]
        
        # Remove old tracks
        self.person_tracks = current_persons
    
    def _update_object_tracks(self, detections: List[Dict[str, Any]], frame_number: int):
        """Update tracking information for objects (non-person)"""
        current_objects = {}
        
        for detection in detections:
            if detection['class_id'] != 0:  # Non-person objects
                bbox = detection['bbox']
                center = detection['center']
                class_name = detection['class_name']
                
                # Find matching track using IoU
                matched_track_id = None
                for track_id, track in self.object_tracks.items():
                    if (track['class_name'] == class_name and 
                        self._calculate_iou(bbox, track['last_bbox']) > 0.3):
                        matched_track_id = track_id
                        break
                
                if matched_track_id is not None:
                    # Update existing track
                    track = self.object_tracks[matched_track_id]
                    track['frames_present'].append(frame_number)
                    track['last_bbox'] = bbox
                    track['last_seen'] = frame_number
                    
                    current_objects[matched_track_id] = track
                else:
                    # Create new track
                    track_id = len(self.object_tracks)
                    self.object_tracks[track_id] = {
                        'class_name': class_name,
                        'frames_present': [frame_number],
                        'last_bbox': bbox,
                        'last_seen': frame_number,
                        'start_frame': frame_number
                    }
                    current_objects[track_id] = self.object_tracks[track_id]
        
        # Remove old tracks
        self.object_tracks = current_objects
    
    def _detect_loitering(self, frame_number: int) -> List[Dict[str, Any]]:
        """Detect loitering behavior"""
        anomalies = []
        
        for track_id, track in self.person_tracks.items():
            # Check if person has been in the same area for too long
            if len(track['frames_present']) >= self.loitering_threshold:
                # Calculate area of movement
                positions = np.array(track['positions'])
                if len(positions) > 1:
                    # Calculate bounding box of movement
                    min_x, min_y = np.min(positions, axis=0)
                    max_x, max_y = np.max(positions, axis=0)
                    area = (max_x - min_x) * (max_y - min_y)
                    
                    # If movement area is small, consider it loitering
                    if area < 10000:  # Threshold for small movement area
                        anomaly = {
                            'type': 'loitering',
                            'track_id': track_id,
                            'frame_number': frame_number,
                            'timestamp': time.time(),
                            'severity': 'medium',
                            'description': f'Person loitering in area for {len(track["frames_present"])} frames',
                            'bbox': track['last_bbox'],
                            'movement_area': area
                        }
                        anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_unusual_movements(self, frame_number: int) -> List[Dict[str, Any]]:
        """Detect unusual movement patterns"""
        anomalies = []
        
        for track_id, track in self.person_tracks.items():
            if len(track['positions']) >= 10:  # Need enough data points
                # Calculate movement features
                positions = np.array(track['positions'])
                velocities = self._calculate_velocities(positions)
                
                if len(velocities) > 0:
                    # Calculate movement statistics
                    avg_velocity = np.mean(velocities)
                    velocity_std = np.std(velocities)
                    
                    # Detect sudden changes in velocity
                    if len(velocities) >= 2:
                        velocity_changes = np.diff(velocities)
                        max_change = np.max(np.abs(velocity_changes))
                        
                        if max_change > self.movement_threshold * 2:
                            anomaly = {
                                'type': 'unusual_movement',
                                'track_id': track_id,
                                'frame_number': frame_number,
                                'timestamp': time.time(),
                                'severity': 'high',
                                'description': f'Sudden movement change detected: {max_change:.3f}',
                                'bbox': track['last_bbox'],
                                'velocity_change': max_change
                            }
                            anomalies.append(anomaly)
                    
                    # Detect erratic movement patterns
                    if velocity_std > avg_velocity * 2:
                        anomaly = {
                            'type': 'erratic_movement',
                            'track_id': track_id,
                            'frame_number': frame_number,
                            'timestamp': time.time(),
                            'severity': 'medium',
                            'description': f'Erratic movement pattern detected',
                            'bbox': track['last_bbox'],
                            'velocity_std': velocity_std,
                            'avg_velocity': avg_velocity
                        }
                        anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_object_abandonment(self, frame_number: int) -> List[Dict[str, Any]]:
        """Detect abandoned objects"""
        anomalies = []
        
        for track_id, track in self.object_tracks.items():
            # Check if object has been stationary for too long
            if (frame_number - track['last_seen']) >= self.abandonment_threshold:
                anomaly = {
                    'type': 'object_abandonment',
                    'track_id': track_id,
                    'frame_number': frame_number,
                    'timestamp': time.time(),
                    'severity': 'medium',
                    'description': f'Object {track["class_name"]} appears to be abandoned',
                    'bbox': track['last_bbox'],
                    'frames_stationary': frame_number - track['last_seen'],
                    'object_class': track['class_name']
                }
                anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_movement(self, positions: List[List[int]]) -> float:
        """Calculate total movement distance"""
        if len(positions) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_distance += math.sqrt(dx*dx + dy*dy)
        
        return total_distance
    
    def _calculate_velocities(self, positions: np.ndarray) -> List[float]:
        """Calculate velocities between consecutive positions"""
        if len(positions) < 2:
            return []
        
        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            velocity = math.sqrt(dx*dx + dy*dy)
            velocities.append(velocity)
        
        return velocities
    
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
    
    def _add_anomaly(self, anomaly: Dict[str, Any]):
        """Add anomaly to history"""
        self.anomaly_history.append(anomaly)
        
        # Keep only recent history
        if len(self.anomaly_history) > self.max_history:
            self.anomaly_history.pop(0)
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of detected anomalies"""
        if not self.anomaly_history:
            return {}
        
        anomaly_types = {}
        severity_counts = {}
        
        for anomaly in self.anomaly_history:
            # Count by type
            anomaly_type = anomaly['type']
            anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1
            
            # Count by severity
            severity = anomaly['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_anomalies': len(self.anomaly_history),
            'anomaly_types': anomaly_types,
            'severity_distribution': severity_counts,
            'recent_anomalies': self.anomaly_history[-10:] if len(self.anomaly_history) >= 10 else self.anomaly_history
        }
    
    def get_tracking_stats(self) -> Dict[str, Any]:
        """Get statistics about current tracking"""
        return {
            'active_person_tracks': len(self.person_tracks),
            'active_object_tracks': len(self.object_tracks),
            'total_anomalies_detected': len(self.anomaly_history)
        }
