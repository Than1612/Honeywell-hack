#!/usr/bin/env python3
"""
AI-Powered Surveillance Dashboard
Upload videos and process them with YOLOv8 model for anomaly detection
"""

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import logging
from datetime import datetime
import json
from pathlib import Path
from ultralytics import YOLO
import time
from synthetic_video_generator import SyntheticVideoGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Configuration
UPLOAD_FOLDER = 'uploads'
SCREENSHOTS_FOLDER = 'screenshots'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MODEL_PATH = 'models/yolov8n_ucsd_quick_test/weights/best.pt'  # Use custom trained model

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SCREENSHOTS_FOLDER, exist_ok=True)

# Global variables
alerts = []
processing_status = "idle"
current_video = None
model = None

# Initialize synthetic video generator
synthetic_generator = SyntheticVideoGenerator()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load YOLO model with fallback"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading custom trained model: {MODEL_PATH}")
            model = YOLO(MODEL_PATH)
        else:
            logger.warning(f"Custom model not found: {MODEL_PATH}")
            logger.info("Loading default YOLOv8n model...")
            model = YOLO('yolov8n.pt')
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def save_screenshot(frame, frame_count):
    """Save screenshot of the frame"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}_frame_{frame_count}.jpg"
    filepath = os.path.join(SCREENSHOTS_FOLDER, filename)
    cv2.imwrite(filepath, frame)
    return filename

def format_time(timestamp):
    """Format timestamp for display"""
    return timestamp.strftime("%H:%M:%S")

def determine_alert_type(detections, frame_count):
    """Determine the type of alert based on detection patterns"""
    if len(detections) > 2:
        return "Multiple Persons Detected"
    elif len(detections) == 1:
        if frame_count < 100:
            return "Person Entry Detected"
        else:
            return "Person Movement Detected"
    else:
        return "Activity Detected"

def generate_alert_description(detections, frame_count, timestamp):
    """Generate descriptive alert message for hackathon demo"""
    num_detections = len(detections)
    avg_confidence = np.mean([d['confidence'] for d in detections])
    
    base_desc = "Single person detected" if num_detections == 1 else f"{num_detections} persons detected simultaneously"
    conf_desc = "with very high confidence" if avg_confidence > 0.9 else ("with high confidence" if avg_confidence > 0.8 else ("with good confidence" if avg_confidence > 0.7 else "with moderate confidence"))
    time_desc = f"at {format_time(timestamp)}"
    frame_desc = " (early video sequence)" if frame_count < 100 else (" (late video sequence)" if frame_count > 500 else " (mid video sequence)")
    
    description = f"{base_desc} {conf_desc} {time_desc}{frame_desc}. "
    
    if num_detections > 1:
        description += "Multiple individuals in frame suggest potential crowd activity or group movement."
    else:
        description += "Individual person detection indicates normal surveillance activity or potential anomaly requiring attention."
    
    return description

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('video_dashboard.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload"""
    global current_video, processing_status
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        current_video = filepath
        processing_status = "uploaded"
        
        logger.info(f"Video uploaded: {filepath}")
        return jsonify({
            'message': 'Video uploaded successfully',
            'filename': filename,
            'status': 'uploaded'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/process', methods=['POST'])
def process_video():
    """Process uploaded video with YOLO model"""
    global alerts, processing_status, current_video
    
    if not current_video or not os.path.exists(current_video):
        return jsonify({'error': 'No video to process'}), 400
    
    if not model:
        return jsonify({'error': 'Model not loaded'}), 400
    
    try:
        processing_status = "processing"
        alerts = []  # Clear previous alerts
        
        # Open video
        cap = cv2.VideoCapture(current_video)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video'}), 400
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Processing video: {total_frames} frames at {fps} FPS")
        
        frame_count = 0
        last_alert_time = 0
        alert_cooldown = 30  # frames between alerts
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run YOLO detection
            results = model(frame, verbose=False)
            
            # Process detections
            high_conf_detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Filter for person class (class 0) with high confidence
                        if box.cls == 0 and box.conf > 0.5:  # Person class with >50% confidence
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0])
                            
                            detection = {
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': confidence,
                                'class': 'person'
                            }
                            high_conf_detections.append(detection)
                            
                            # Draw bounding box on frame
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, f'Person: {confidence:.2f}', 
                                       (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Generate alerts for high-confidence detections
            if len(high_conf_detections) > 0:
                current_time = datetime.now()
                
                if (len(high_conf_detections) > 0 and 
                    frame_count - last_alert_time > alert_cooldown):
                    try:
                        # Save screenshot
                        screenshot_path = save_screenshot(frame, frame_count)
                        
                        # Generate descriptive alert message
                        alert_description = generate_alert_description(high_conf_detections, frame_count, current_time)
                        
                        # Create alert
                        alert = {
                            'id': len(alerts) + 1,
                            'timestamp': current_time,
                            'frame': frame_count,
                            'detections': len(high_conf_detections),
                            'confidence': float(np.mean([d['confidence'] for d in high_conf_detections])),
                            'screenshot': screenshot_path,
                            'details': high_conf_detections,
                            'time_formatted': format_time(current_time),
                            'description': alert_description,
                            'alert_type': determine_alert_type(high_conf_detections, frame_count)
                        }
                        
                        alerts.append(alert)
                        last_alert_time = frame_count
                        
                        logger.info(f"Alert at frame {frame_count}: {len(high_conf_detections)} detections")
                    except Exception as e:
                        logger.error(f"Error creating alert at frame {frame_count}: {e}")
                        continue # Continue processing even if alert creation fails
        
        cap.release()
        processing_status = "completed"
        
        logger.info(f"Video processing completed. Generated {len(alerts)} alerts.")
        
        return jsonify({
            'message': 'Video processed successfully',
            'alerts_count': len(alerts),
            'status': 'completed'
        })
        
    except Exception as e:
        processing_status = "failed"
        logger.error(f"Error processing video: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/status')
def get_status():
    """Get current processing status and alerts"""
    return jsonify({
        'status': processing_status,
        'current_video': current_video,
        'alerts_count': len(alerts),
        'alerts': alerts
    })

@app.route('/screenshots/<filename>')
def get_screenshot(filename):
    """Serve screenshot files"""
    return send_from_directory(SCREENSHOTS_FOLDER, filename)

@app.route('/generate_synthetic', methods=['POST'])
def generate_synthetic_video():
    """Generate synthetic video for edge case testing"""
    try:
        data = request.get_json()
        scenario = data.get('scenario', 'all')
        
        logger.info(f"Generating synthetic video for scenario: {scenario}")
        
        if scenario == 'all':
            results = synthetic_generator.generate_all_scenarios()
            if results:
                return jsonify({
                    'message': 'All synthetic videos generated successfully',
                    'scenarios': results,
                    'status': 'success'
                })
            else:
                return jsonify({'error': 'Failed to generate synthetic videos'}), 500
        else:
            # Generate specific scenario
            if scenario == 'loitering':
                path = synthetic_generator.generate_loitering_scenario()
            elif scenario == 'crowd_surge':
                path = synthetic_generator.generate_crowd_surge_scenario()
            elif scenario == 'rapid_movement':
                path = synthetic_generator.generate_rapid_movement_scenario()
            elif scenario == 'low_light':
                path = synthetic_generator.generate_low_light_scenario()
            else:
                return jsonify({'error': 'Invalid scenario'}), 400
            
            return jsonify({
                'message': f'{scenario} scenario generated successfully',
                'video_path': path,
                'status': 'success'
            })
            
    except Exception as e:
        logger.error(f"Error generating synthetic video: {e}")
        return jsonify({'error': f'Generation failed: {str(e)}'}), 500

@app.route('/list_synthetic')
def list_synthetic_videos():
    """List all generated synthetic videos"""
    try:
        videos = synthetic_generator.list_generated_videos()
        return jsonify({
            'videos': videos,
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Error listing synthetic videos: {e}")
        return jsonify({'error': f'Failed to list videos: {str(e)}'}), 500

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("Starting dashboard...")
        app.run(host='0.0.0.0', port=5001, debug=True)
    else:
        logger.error("Failed to load model. Dashboard cannot start.")
