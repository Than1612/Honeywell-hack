"""
Configuration file for AI Surveillance System
Contains all configurable parameters and settings
"""

import os
from typing import Dict, Any

class Config:
    """Configuration class for surveillance system"""
    
    # System Configuration
    SYSTEM_NAME = "AI-Powered Surveillance System"
    VERSION = "1.0.0"
    DEBUG = True
    
    # Video Processing
    VIDEO_CONFIG = {
        'default_fps': 30,
        'default_resolution': (640, 480),
        'max_frame_size': (1920, 1080),
        'min_frame_size': (320, 240),
        'frame_buffer_size': 100
    }
    
    # Object Detection
    OBJECT_DETECTION_CONFIG = {
        'model_path': 'yolov8n.pt',
        'confidence_threshold': 0.5,
        'nms_threshold': 0.4,
        'max_detections': 100,
        'device': 'auto',  # 'cpu', 'cuda', or 'auto'
        'classes': None,  # None for all classes, or list of class IDs
        'input_size': (640, 640)
    }
    
    # Anomaly Detection
    ANOMALY_DETECTION_CONFIG = {
        'loitering_threshold': 50,  # frames
        'movement_threshold': 0.1,
        'abandonment_threshold': 100,  # frames
        'tracking_history_length': 30,
        'iou_threshold': 0.3,
        'min_track_length': 5,
        'anomaly_cooldown': 10  # frames between same anomaly
    }
    
    # Dashboard Configuration
    DASHBOARD_CONFIG = {
        'host': '0.0.0.0',
        'port': 5000,
        'update_interval': 1.0,  # seconds
        'max_alerts': 100,
        'max_history': 1000,
        'enable_cors': True
    }
    
    # Output Configuration
    OUTPUT_CONFIG = {
        'save_video': True,
        'save_screenshots': True,
        'output_directory': 'output',
        'video_format': 'mp4v',
        'image_format': 'jpg',
        'max_output_size': 1024 * 1024 * 1024  # 1GB
    }
    
    # Logging Configuration
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_handler': True,
        'console_handler': True,
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5
    }
    
    # Performance Configuration
    PERFORMANCE_CONFIG = {
        'max_threads': 4,
        'batch_size': 1,
        'enable_gpu': True,
        'memory_limit': 0.8,  # 80% of available memory
        'processing_timeout': 30.0  # seconds
    }
    
    # Alert Configuration
    ALERT_CONFIG = {
        'enable_email_alerts': False,
        'enable_sms_alerts': False,
        'enable_push_notifications': False,
        'alert_cooldown': 300,  # seconds
        'max_alerts_per_hour': 100,
        'alert_retention_days': 30
    }
    
    # Security Configuration
    SECURITY_CONFIG = {
        'enable_authentication': False,
        'enable_encryption': False,
        'allowed_ips': ['127.0.0.1', 'localhost'],
        'max_login_attempts': 3,
        'session_timeout': 3600  # seconds
    }
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        config = {}
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                value = getattr(cls, attr)
                if not callable(value):
                    config[attr] = value
        return config
    
    @classmethod
    def update_config(cls, updates: Dict[str, Any]):
        """Update configuration with new values"""
        for key, value in updates.items():
            if hasattr(cls, key) and not key.startswith('_'):
                setattr(cls, key, value)
    
    @classmethod
    def load_from_file(cls, config_file: str):
        """Load configuration from JSON file"""
        if os.path.exists(config_file):
            import json
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                cls.update_config(config_data)
                print(f"Configuration loaded from {config_file}")
            except Exception as e:
                print(f"Error loading configuration: {e}")
    
    @classmethod
    def save_to_file(cls, config_file: str):
        """Save current configuration to JSON file"""
        import json
        try:
            config_data = cls.get_config()
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            print(f"Configuration saved to {config_file}")
        except Exception as e:
            print(f"Error saving configuration: {e}")

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    LOGGING_CONFIG = {
        'level': 'DEBUG',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_handler': True,
        'console_handler': True,
        'max_file_size': 5 * 1024 * 1024,  # 5MB
        'backup_count': 3
    }
    PERFORMANCE_CONFIG = {
        'max_threads': 2,
        'batch_size': 1,
        'enable_gpu': False,
        'memory_limit': 0.5,
        'processing_timeout': 60.0
    }

class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    LOGGING_CONFIG = {
        'level': 'WARNING',
        'format': '%(asctime)s - %(levelname)s - %(message)s',
        'file_handler': True,
        'console_handler': False,
        'max_file_size': 50 * 1024 * 1024,  # 50MB
        'backup_count': 10
    }
    PERFORMANCE_CONFIG = {
        'max_threads': 8,
        'batch_size': 4,
        'enable_gpu': True,
        'memory_limit': 0.9,
        'processing_timeout': 15.0
    }
    ALERT_CONFIG = {
        'enable_email_alerts': True,
        'enable_sms_alerts': True,
        'enable_push_notifications': True,
        'alert_cooldown': 60,
        'max_alerts_per_hour': 1000,
        'alert_retention_days': 90
    }

# Configuration factory
def get_config(environment: str = 'development') -> Config:
    """Get configuration for specified environment"""
    if environment.lower() == 'production':
        return ProductionConfig
    else:
        return DevelopmentConfig

# Default configuration instance
config = Config()

if __name__ == '__main__':
    # Print current configuration
    print("Current Configuration:")
    for key, value in config.get_config().items():
        print(f"{key}: {value}")
    
    # Save configuration to file
    config.save_to_file('config.json')
