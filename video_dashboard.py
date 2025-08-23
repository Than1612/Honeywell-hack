#!/usr/bin/env python3
"""
Video Processing Dashboard
Simple dashboard for processing videos with your trained UCSD model
"""

import os
import cv2
import numpy as np
import base64
from datetime import datetime
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import logging
from ultralytics import YOLO
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['PROCESSED_FOLDER'] = 'processed_videos'
app.config['SCREENSHOTS_FOLDER'] = 'screenshots'

# Create directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER'], app.config['SCREENSHOTS_FOLDER']]:
    Path(folder).mkdir(exist_ok=True)

# Global variables
current_video = None
processing_status = "idle"
alerts = []
model = None

def load_model():
    """Load the trained UCSD model"""
    global model
    try:
        # Try to load the trained model
        model_paths = [
            "models/yolov8n_ucsd_quick/weights/best.pt",
            "models/yolov8n_ucsd_quick_test/weights/best.pt",
            "yolov8n.pt"  # Fallback
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                model = YOLO(path)
                logger.info(f"Model loaded from: {path}")
                return True
        
        logger.error("No model found!")
        return False
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def process_video(video_path):
    """Process video with YOLOv8 model and detect anomalies"""
    global alerts, processing_status
    
    alerts = []
    processing_status = "processing"
    
    try:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_count = 0
        last_alert_time = 0
        alert_cooldown = fps * 2  # 2 seconds between alerts
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = frame_count / fps
            
            # Run detection every 5 frames for performance
            if frame_count % 5 == 0:
                results = model(frame, verbose=False)
                
                # Check for detections
                if len(results) > 0:
                    result = results[0]
                    if result.boxes is not None and len(result.boxes) > 0:
                        # Get detection info
                        boxes = result.boxes
                        confidences = boxes.conf.cpu().numpy()
                        class_ids = boxes.cls.cpu().numpy()
                        
                        # Check for high-confidence detections
                        high_conf_detections = []
                        for i, (conf, class_id) in enumerate(zip(confidences, class_ids)):
                            if conf > 0.5:  # Confidence threshold
                                high_conf_detections.append({
                                    'confidence': float(conf),
                                    'class': int(class_id),
                                    'box': boxes.xyxy[i].cpu().numpy().tolist()
                                })
                        
                                                                            # Create alert if significant detections found
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
                                continue
            
            # Update progress
            if frame_count % 30 == 0:  # Every 30 frames
                progress = (frame_count / total_frames) * 100
                logger.info(f"Processing: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        processing_status = "completed"
        
        # Save alerts to file
        save_alerts()
        
        logger.info(f"Video processing completed. {len(alerts)} alerts generated.")
        return True
        
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        processing_status = "failed"
        return False

def save_screenshot(frame, frame_count):
    """Save screenshot of the frame"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alert_frame_{frame_count}_{timestamp}.jpg"
        filepath = os.path.join(app.config['SCREENSHOTS_FOLDER'], filename)
        
        # Resize frame for storage
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        cv2.imwrite(filepath, frame)
        return filename
        
    except Exception as e:
        logger.error(f"Failed to save screenshot: {e}")
        return None

def save_alerts():
    """Save alerts to JSON file"""
    try:
        alerts_file = os.path.join(app.config['PROCESSED_FOLDER'], 'alerts.json')
        with open(alerts_file, 'w') as f:
            json.dump(alerts, f, indent=2, default=str)
        logger.info(f"Alerts saved to: {alerts_file}")
    except Exception as e:
        logger.error(f"Failed to save alerts: {e}")

def format_time(seconds):
    """Format time in seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def determine_alert_type(detections, frame_count):
    """Determine the type of alert based on detection patterns"""
    if len(detections) > 2:
        return "Multiple Persons Detected"
    elif len(detections) == 1:
        # Check if this is early in video (potential intrusion)
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
    
    # Base description
    if num_detections == 1:
        base_desc = "Single person detected"
    else:
        base_desc = f"{num_detections} persons detected simultaneously"
    
    # Confidence level description
    if avg_confidence > 0.9:
        conf_desc = "with very high confidence"
    elif avg_confidence > 0.8:
        conf_desc = "with high confidence"
    elif avg_confidence > 0.7:
        conf_desc = "with good confidence"
    else:
        conf_desc = "with moderate confidence"
    
    # Time context
    time_desc = f"at {format_time(timestamp)}"
    
    # Frame context
    if frame_count < 100:
        frame_desc = " (early video sequence)"
    elif frame_count > 500:
        frame_desc = " (late video sequence)"
    else:
        frame_desc = " (mid video sequence)"
    
    # Combine into descriptive message
    description = f"{base_desc} {conf_desc} {time_desc}{frame_desc}. "
    
    # Add situational context
    if num_detections > 1:
        description += "Multiple individuals in frame suggest potential crowd activity or group movement."
    else:
        description += "Individual person detection indicates normal surveillance activity or potential anomaly requiring attention."
    
    return description

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('video_dashboard.html', 
                         alerts=alerts, 
                         status=processing_status)

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload"""
    global current_video
    
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(filepath)
            current_video = filepath
            
            logger.info(f"Video uploaded: {filename}")
            return jsonify({'success': True, 'filename': filename})
            
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def start_processing():
    """Start video processing"""
    global current_video
    
    if not current_video:
        return jsonify({'error': 'No video uploaded'}), 400
    
    if not model:
        return jsonify({'error': 'Model not loaded'}), 400
    
    try:
        # Start processing in background (simple approach for hackathon)
        success = process_video(current_video)
        
        if success:
            return jsonify({
                'success': True, 
                'message': f'Processing completed. {len(alerts)} alerts generated.',
                'alerts_count': len(alerts)
            })
        else:
            return jsonify({'error': 'Processing failed'}), 500
            
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def get_status():
    """Get current processing status"""
    return jsonify({
        'status': processing_status,
        'alerts_count': len(alerts),
        'current_video': current_video
    })

@app.route('/alerts')
def get_alerts():
    """Get all alerts"""
    return jsonify(alerts)

@app.route('/screenshot/<filename>')
def get_screenshot(filename):
    """Serve screenshot files"""
    try:
        return send_file(os.path.join(app.config['SCREENSHOTS_FOLDER'], filename))
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/clear', methods=['POST'])
def clear_data():
    """Clear all data"""
    global alerts, current_video, processing_status
    
    alerts = []
    current_video = None
    processing_status = "idle"
    
    # Clear screenshots
    for file in os.listdir(app.config['SCREENSHOTS_FOLDER']):
        if file.endswith('.jpg'):
            os.remove(os.path.join(app.config['SCREENSHOTS_FOLDER'], file))
    
    logger.info("Data cleared")
    return jsonify({'success': True, 'message': 'Data cleared'})

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("🚀 Video Dashboard ready!")
        logger.info("Model loaded successfully")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        logger.error("❌ Failed to load model. Dashboard cannot start.")
        exit(1)
