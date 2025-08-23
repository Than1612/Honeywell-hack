#!/usr/bin/env python3
"""
Simple runner script for the AI Surveillance System
"""

import sys
import os
import subprocess
import time
import logging

def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'ultralytics',
        'opencv-python',
        'numpy',
        'torch',
        'flask',
        'scikit-learn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Install them using: pip install -r requirements.txt")
        return False
    
    return True

def start_dashboard():
    """Start the dashboard in background"""
    try:
        print("Starting dashboard...")
        dashboard_process = subprocess.Popen([
            sys.executable, 'dashboard.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for dashboard to start
        time.sleep(3)
        
        if dashboard_process.poll() is None:
            print("Dashboard started successfully")
            return dashboard_process
        else:
            print("Failed to start dashboard")
            return None
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        return None

def start_surveillance(video_source="0"):
    """Start the surveillance system"""
    try:
        print(f"Starting surveillance system with source: {video_source}")
        surveillance_process = subprocess.Popen([
            sys.executable, 'main.py', '--source', video_source
        ])
        
        return surveillance_process
    except Exception as e:
        print(f"Error starting surveillance: {e}")
        return None

def main():
    """Main function"""
    setup_logging()
    
    print("=" * 50)
    print("AI-Powered Surveillance System")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Get video source
    if len(sys.argv) > 1:
        video_source = sys.argv[1]
    else:
        video_source = input("Enter video source (0 for webcam, or file path): ").strip()
        if not video_source:
            video_source = "0"
    
    # Start dashboard
    dashboard_process = start_dashboard()
    if not dashboard_process:
        print("Continuing without dashboard...")
    
    # Start surveillance
    surveillance_process = start_surveillance(video_source)
    if not surveillance_process:
        print("Failed to start surveillance system")
        if dashboard_process:
            dashboard_process.terminate()
        sys.exit(1)
    
    try:
        print("\nSurveillance system is running!")
        print("Press Ctrl+C to stop")
        print(f"Dashboard available at: http://localhost:5000")
        
        # Wait for processes
        while True:
            if surveillance_process.poll() is not None:
                print("Surveillance system stopped")
                break
            if dashboard_process and dashboard_process.poll() is not None:
                print("Dashboard stopped")
                break
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping system...")
        
        # Terminate processes
        if surveillance_process:
            surveillance_process.terminate()
        if dashboard_process:
            dashboard_process.terminate()
        
        print("System stopped")

if __name__ == '__main__':
    main()
