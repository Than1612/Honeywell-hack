from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import time
from datetime import datetime
import threading
import logging
from typing import Dict, Any, List
import os

class SurveillanceDashboard:
    """
    Flask-based web dashboard for surveillance system
    """
    
    def __init__(self, host: str = '0.0.0.0', port: int = 5000):
        self.app = Flask(__name__)
        CORS(self.app)
        self.host = host
        self.port = port
        
        # Dashboard data
        self.live_data = {
            'current_frame': None,
            'detections': [],
            'anomalies': [],
            'stats': {},
            'alerts': []
        }
        
        # Alert history
        self.alert_history = []
        self.max_alerts = 100
        
        # Setup routes
        self._setup_routes()
        
        logging.info(f"Dashboard initialized on {host}:{port}")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/api/live-data')
        def get_live_data():
            return jsonify(self.live_data)
        
        @self.app.route('/api/alerts')
        def get_alerts():
            return jsonify({
                'current_alerts': self.live_data['alerts'],
                'alert_history': self.alert_history[-20:]  # Last 20 alerts
            })
        
        @self.app.route('/api/stats')
        def get_stats():
            return jsonify(self.live_data['stats'])
        
        @self.app.route('/api/detections')
        def get_detections():
            return jsonify(self.live_data['detections'])
        
        @self.app.route('/api/anomalies')
        def get_anomalies():
            return jsonify(self.live_data['anomalies'])
        
        @self.app.route('/api/update', methods=['POST'])
        def update_data():
            """Update dashboard data from surveillance system"""
            try:
                data = request.json
                self.update_dashboard_data(data)
                return jsonify({'status': 'success'})
            except Exception as e:
                logging.error(f"Error updating dashboard: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def update_dashboard_data(self, data: Dict[str, Any]):
        """Update dashboard with new data from surveillance system"""
        if 'detections' in data:
            self.live_data['detections'] = data['detections']
        
        if 'anomalies' in data:
            new_anomalies = data['anomalies']
            self.live_data['anomalies'] = new_anomalies
            
            # Process new anomalies as alerts
            for anomaly in new_anomalies:
                if anomaly not in self.live_data['alerts']:
                    self._add_alert(anomaly)
        
        if 'stats' in data:
            self.live_data['stats'] = data['stats']
        
        if 'current_frame' in data:
            self.live_data['current_frame'] = data['current_frame']
    
    def _add_alert(self, anomaly: Dict[str, Any]):
        """Add new anomaly as alert"""
        alert = {
            'id': len(self.alert_history) + 1,
            'timestamp': datetime.now().isoformat(),
            'type': anomaly.get('type', 'unknown'),
            'severity': anomaly.get('severity', 'medium'),
            'description': anomaly.get('description', 'Anomaly detected'),
            'bbox': anomaly.get('bbox', []),
            'frame_number': anomaly.get('frame_number', 0)
        }
        
        self.live_data['alerts'].append(alert)
        self.alert_history.append(alert)
        
        # Keep only recent alerts
        if len(self.live_data['alerts']) > 10:
            self.live_data['alerts'].pop(0)
        
        if len(self.alert_history) > self.max_alerts:
            self.alert_history.pop(0)
    
    def start(self):
        """Start the dashboard server"""
        try:
            self.app.run(host=self.host, port=self.port, debug=False, threaded=True)
        except Exception as e:
            logging.error(f"Failed to start dashboard: {e}")
            raise

# Create templates directory and HTML template
def create_dashboard_template():
    """Create the HTML template for the dashboard"""
    os.makedirs('templates', exist_ok=True)
    
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Surveillance Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .video-container {
            background: #000;
            border-radius: 10px;
            overflow: hidden;
            min-height: 400px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }
        .stat-item {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            font-size: 14px;
        }
        .alerts-container {
            max-height: 400px;
            overflow-y: auto;
        }
        .alert-item {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }
        .alert-high { border-color: #dc3545; background: #f8d7da; }
        .alert-medium { border-color: #ffc107; background: #fff3cd; }
        .alert-low { border-color: #28a745; background: #d4edda; }
        .refresh-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin-bottom: 20px;
        }
        .refresh-btn:hover {
            background: #5a6fd8;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚨 AI-Powered Surveillance Dashboard</h1>
        <p>Real-time monitoring and anomaly detection</p>
    </div>
    
    <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>
    
    <div class="dashboard-grid">
        <div class="card">
            <h3>📹 Live Video Feed</h3>
            <div class="video-container" id="videoContainer">
                <div>Video feed will appear here</div>
            </div>
        </div>
        
        <div class="card">
            <h3>📊 System Statistics</h3>
            <div class="stats-grid" id="statsGrid">
                <div class="stat-item">
                    <div class="stat-value" id="totalDetections">0</div>
                    <div class="stat-label">Total Detections</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="activeTracks">0</div>
                    <div class="stat-label">Active Tracks</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="totalAnomalies">0</div>
                    <div class="stat-label">Anomalies Detected</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="fps">0</div>
                    <div class="stat-label">FPS</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="dashboard-grid">
        <div class="card">
            <h3>🎯 Recent Detections</h3>
            <div id="detectionsContainer">
                <p>No detections available</p>
            </div>
        </div>
        
        <div class="card">
            <h3>🚨 Active Alerts</h3>
            <div class="alerts-container" id="alertsContainer">
                <p>No active alerts</p>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h3>📈 Anomaly Analysis</h3>
        <div id="anomalyChart"></div>
    </div>

    <script>
        let updateInterval;
        
        function refreshData() {
            fetchLiveData();
            fetchAlerts();
            fetchStats();
        }
        
        function fetchLiveData() {
            fetch('/api/live-data')
                .then(response => response.json())
                .then(data => {
                    updateDetections(data.detections);
                    updateStats(data.stats);
                })
                .catch(error => console.error('Error fetching live data:', error));
        }
        
        function fetchAlerts() {
            fetch('/api/alerts')
                .then(response => response.json())
                .then(data => {
                    updateAlerts(data.current_alerts);
                })
                .catch(error => console.error('Error fetching alerts:', error));
        }
        
        function fetchStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    updateStats(data);
                })
                .catch(error => console.error('Error fetching stats:', error));
        }
        
        function updateDetections(detections) {
            const container = document.getElementById('detectionsContainer');
            if (!detections || detections.length === 0) {
                container.innerHTML = '<p>No detections available</p>';
                return;
            }
            
            let html = '<div style="max-height: 300px; overflow-y: auto;">';
            detections.forEach(det => {
                html += `
                    <div style="padding: 8px; margin: 5px 0; background: #f8f9fa; border-radius: 5px;">
                        <strong>${det.class_name}</strong> (${(det.confidence * 100).toFixed(1)}%)
                        <br><small>Position: [${det.center[0]}, ${det.center[1]}]</small>
                    </div>
                `;
            });
            html += '</div>';
            container.innerHTML = html;
        }
        
        function updateAlerts(alerts) {
            const container = document.getElementById('alertsContainer');
            if (!alerts || alerts.length === 0) {
                container.innerHTML = '<p>No active alerts</p>';
                return;
            }
            
            let html = '';
            alerts.forEach(alert => {
                const severityClass = `alert-${alert.severity}`;
                html += `
                    <div class="alert-item ${severityClass}">
                        <strong>${alert.type.toUpperCase()}</strong><br>
                        ${alert.description}<br>
                        <small>${new Date(alert.timestamp).toLocaleTimeString()}</small>
                    </div>
                `;
            });
            container.innerHTML = html;
        }
        
        function updateStats(stats) {
            if (stats.total_detections !== undefined) {
                document.getElementById('totalDetections').textContent = stats.total_detections;
            }
            if (stats.active_person_tracks !== undefined) {
                document.getElementById('activeTracks').textContent = stats.active_person_tracks;
            }
            if (stats.total_anomalies_detected !== undefined) {
                document.getElementById('totalAnomalies').textContent = stats.total_anomalies_detected;
            }
            if (stats.fps !== undefined) {
                document.getElementById('fps').textContent = stats.fps.toFixed(1);
            }
        }
        
        function createAnomalyChart() {
            const data = [
                {
                    x: ['Loitering', 'Unusual Movement', 'Object Abandonment'],
                    y: [0, 0, 0],
                    type: 'bar',
                    marker: {
                        color: ['#ffc107', '#dc3545', '#28a745']
                    }
                }
            ];
            
            const layout = {
                title: 'Anomaly Distribution',
                xaxis: { title: 'Anomaly Type' },
                yaxis: { title: 'Count' }
            };
            
            Plotly.newPlot('anomalyChart', data, layout);
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            createAnomalyChart();
            refreshData();
            
            // Auto-refresh every 2 seconds
            updateInterval = setInterval(refreshData, 2000);
        });
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', function() {
            if (updateInterval) {
                clearInterval(updateInterval);
            }
        });
    </script>
</body>
</html>'''
    
    with open('templates/index.html', 'w') as f:
        f.write(html_template)

if __name__ == '__main__':
    # Create template
    create_dashboard_template()
    
    # Start dashboard
    dashboard = SurveillanceDashboard()
    dashboard.start()
