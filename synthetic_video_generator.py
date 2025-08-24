#!/usr/bin/env python3
"""
Synthetic Video Generator for Edge Cases
Creates realistic surveillance videos with various anomalies for testing
"""

import cv2
import numpy as np
import random
from pathlib import Path
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticVideoGenerator:
    """Generate synthetic surveillance videos with edge cases"""
    
    def __init__(self, output_dir: str = "synthetic_videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Video parameters
        self.width = 1920
        self.height = 1080
        self.fps = 30
        self.duration = 10  # seconds
        
        # Edge case scenarios
        self.scenarios = {
            'loitering': 'Person staying in one place for extended time',
            'crowd_surge': 'Multiple people entering simultaneously',
            'rapid_movement': 'Person moving very fast across frame',
            'partial_occlusion': 'Person partially hidden behind objects',
            'low_light': 'Poor lighting conditions',
            'motion_blur': 'Fast movement causing blur',
            'multiple_objects': 'Multiple people with different behaviors',
            'anomalous_pattern': 'Unusual walking patterns'
        }
        
        logger.info(f"Synthetic Video Generator initialized. Output: {self.output_dir}")
    
    def generate_person_sprite(self, person_id: int = 0):
        """Generate a simple person sprite"""
        # Create a simple humanoid shape
        sprite = np.zeros((80, 40, 3), dtype=np.uint8)
        
        # Head (circle)
        cv2.circle(sprite, (20, 15), 8, (255, 200, 150), -1)  # Skin color
        
        # Body (rectangle)
        cv2.rectangle(sprite, (15, 25), (25, 60), (100, 100, 200), -1)  # Blue shirt
        
        # Arms
        cv2.rectangle(sprite, (10, 30), (15, 50), (255, 200, 150), -1)  # Left arm
        cv2.rectangle(sprite, (25, 30), (30, 50), (255, 200, 150), -1)  # Right arm
        
        # Legs
        cv2.rectangle(sprite, (15, 60), (20, 75), (50, 50, 50), -1)   # Left leg
        cv2.rectangle(sprite, (20, 60), (25, 75), (50, 50, 50), -1)   # Right leg
        
        return sprite
    
    def add_noise_and_blur(self, frame, noise_level: float = 0.1, blur_level: int = 1):
        """Add realistic noise and blur to frame"""
        # Add noise
        noise = np.random.normal(0, noise_level * 255, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)
        
        # Add blur
        if blur_level > 0:
            frame = cv2.GaussianBlur(frame, (blur_level * 2 + 1, blur_level * 2 + 1), 0)
        
        return frame
    
    def generate_loitering_scenario(self):
        """Generate video with person loitering in one place"""
        logger.info("Generating loitering scenario...")
        
        video_path = self.output_dir / "loitering_scenario.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, self.fps, (self.width, self.height))
        
        total_frames = self.fps * self.duration
        
        for frame_num in range(total_frames):
            # Create background
            frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 50
            
            # Add some background elements
            cv2.rectangle(frame, (100, 100), (300, 200), (100, 100, 100), -1)  # Building
            cv2.rectangle(frame, (400, 150), (500, 250), (80, 80, 80), -1)     # Another building
            
            # Person sprite
            person = self.generate_person_sprite()
            
            # Person stays in one place (loitering)
            x, y = 800, 400
            frame[y:y+80, x:x+40] = person
            
            # Add timestamp
            cv2.putText(frame, f"Frame: {frame_num}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "LOITERING SCENARIO", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Add realistic effects
            frame = self.add_noise_and_blur(frame, noise_level=0.05)
            
            out.write(frame)
        
        out.release()
        logger.info(f"Loitering scenario saved: {video_path}")
        return str(video_path)
    
    def generate_crowd_surge_scenario(self):
        """Generate video with multiple people entering simultaneously"""
        logger.info("Generating crowd surge scenario...")
        
        video_path = self.output_dir / "crowd_surge_scenario.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, self.fps, (self.width, self.height))
        
        total_frames = self.fps * self.duration
        
        for frame_num in range(total_frames):
            # Create background
            frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 50
            
            # Add background elements
            cv2.rectangle(frame, (0, 0), (self.width, 100), (0, 0, 0), -1)  # Top border
            
            # Multiple people entering from different directions
            people_positions = []
            
            # Person 1: entering from left
            if frame_num > 30:
                x1 = min(800, 200 + frame_num * 2)
                y1 = 300
                people_positions.append((x1, y1, 0))
            
            # Person 2: entering from right
            if frame_num > 45:
                x2 = max(400, 1200 - (frame_num - 45) * 2)
                y2 = 400
                people_positions.append((x2, y2, 1))
            
            # Person 3: entering from top
            if frame_num > 60:
                x3 = 600
                y3 = min(500, 100 + (frame_num - 60) * 2)
                people_positions.append((x3, y3, 2))
            
            # Person 4: entering from bottom
            if frame_num > 75:
                x4 = 900
                y4 = max(300, 800 - (frame_num - 75) * 2)
                people_positions.append((x4, y4, 3))
            
            # Draw all people
            for x, y, person_id in people_positions:
                person = self.generate_person_sprite(person_id)
                frame[y:y+80, x:x+40] = person
            
            # Add labels
            cv2.putText(frame, f"Frame: {frame_num}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"CROWD SURGE: {len(people_positions)} people", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add realistic effects
            frame = self.add_noise_and_blur(frame, noise_level=0.08)
            
            out.write(frame)
        
        out.release()
        logger.info(f"Crowd surge scenario saved: {video_path}")
        return str(video_path)
    
    def generate_rapid_movement_scenario(self):
        """Generate video with person moving very fast"""
        logger.info("Generating rapid movement scenario...")
        
        video_path = self.output_dir / "rapid_movement_scenario.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, self.fps, (self.width, self.height))
        
        total_frames = self.fps * self.duration
        
        for frame_num in range(total_frames):
            # Create background
            frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 50
            
            # Add background elements
            cv2.rectangle(frame, (100, 100), (300, 200), (100, 100, 100), -1)
            
            # Person moving very fast across frame
            speed = 15  # pixels per frame
            x = (frame_num * speed) % (self.width + 100) - 50
            y = 400
            
            if 0 <= x <= self.width - 40:
                person = self.generate_person_sprite()
                frame[y:y+80, x:x+40] = person
                
                # Add motion blur effect
                if x > 0 and x < self.width - 40:
                    # Create motion trail
                    for i in range(1, 4):
                        trail_x = max(0, x - i * 5)
                        trail_person = self.generate_person_sprite()
                        # Make trail more transparent
                        trail_person = (trail_person * (0.7 - i * 0.2)).astype(np.uint8)
                        frame[y:y+80, trail_x:trail_x+40] = trail_person
            
            # Add labels
            cv2.putText(frame, f"Frame: {frame_num}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "RAPID MOVEMENT SCENARIO", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add motion blur
            frame = self.add_noise_and_blur(frame, noise_level=0.06, blur_level=2)
            
            out.write(frame)
        
        out.release()
        logger.info(f"Rapid movement scenario saved: {video_path}")
        return str(video_path)
    
    def generate_low_light_scenario(self):
        """Generate video with poor lighting conditions"""
        logger.info("Generating low light scenario...")
        
        video_path = self.output_dir / "low_light_scenario.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, self.fps, (self.width, self.height))
        
        total_frames = self.fps * self.duration
        
        for frame_num in range(total_frames):
            # Create background
            frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 20  # Dark background
            
            # Add some background elements
            cv2.rectangle(frame, (100, 100), (300, 200), (30, 30, 30), -1)
            
            # Person sprite
            person = self.generate_person_sprite()
            
            # Person moving slowly
            x = 400 + int(50 * np.sin(frame_num * 0.1))
            y = 300 + int(30 * np.cos(frame_num * 0.15))
            
            frame[y:y+80, x:x+40] = person
            
            # Add labels
            cv2.putText(frame, f"Frame: {frame_num}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            cv2.putText(frame, "LOW LIGHT SCENARIO", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Add low light effects
            frame = self.add_noise_and_blur(frame, noise_level=0.15, blur_level=1)
            
            # Make frame darker
            frame = (frame * 0.3).astype(np.uint8)
            
            out.write(frame)
        
        out.release()
        logger.info(f"Low light scenario saved: {video_path}")
        return str(video_path)
    
    def generate_all_scenarios(self):
        """Generate all edge case scenarios"""
        logger.info("Generating all synthetic video scenarios...")
        
        results = {}
        
        try:
            results['loitering'] = self.generate_loitering_scenario()
            results['crowd_surge'] = self.generate_crowd_surge_scenario()
            results['rapid_movement'] = self.generate_rapid_movement_scenario()
            results['low_light'] = self.generate_low_light_scenario()
            
            # Save scenario info
            scenario_info = {
                'generated_at': datetime.now().isoformat(),
                'scenarios': self.scenarios,
                'video_files': results,
                'parameters': {
                    'resolution': f"{self.width}x{self.height}",
                    'fps': self.fps,
                    'duration': self.duration
                }
            }
            
            info_file = self.output_dir / "scenarios_info.json"
            with open(info_file, 'w') as f:
                json.dump(scenario_info, f, indent=2)
            
            logger.info(f"All scenarios generated successfully! Info saved to: {info_file}")
            return results
            
        except Exception as e:
            logger.error(f"Error generating scenarios: {e}")
            return {}
    
    def list_generated_videos(self):
        """List all generated synthetic videos"""
        video_files = list(self.output_dir.glob("*.mp4"))
        
        videos = []
        for video_file in video_files:
            videos.append({
                'name': video_file.name,
                'path': str(video_file),
                'size_mb': video_file.stat().st_size / (1024 * 1024),
                'scenario': video_file.stem.replace('_scenario', '')
            })
        
        return videos


def main():
    """Main function to generate synthetic videos"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Synthetic Surveillance Videos')
    parser.add_argument('--scenario', choices=['loitering', 'crowd_surge', 'rapid_movement', 'low_light', 'all'], 
                       default='all', help='Scenario to generate')
    parser.add_argument('--list', action='store_true', help='List generated videos')
    
    args = parser.parse_args()
    
    generator = SyntheticVideoGenerator()
    
    try:
        if args.list:
            videos = generator.list_generated_videos()
            print("\n" + "="*50)
            print("GENERATED SYNTHETIC VIDEOS")
            print("="*50)
            
            if videos:
                for video in videos:
                    print(f"📹 {video['scenario'].upper()}")
                    print(f"   File: {video['name']}")
                    print(f"   Size: {video['size_mb']:.1f} MB")
                    print(f"   Path: {video['path']}")
                    print()
            else:
                print("No videos generated yet. Run with --scenario to generate videos.")
                
        elif args.scenario == 'all':
            print("🚀 Generating all synthetic video scenarios...")
            results = generator.generate_all_scenarios()
            
            if results:
                print("\n" + "🎉 ALL SCENARIOS GENERATED SUCCESSFULLY!")
                print("="*50)
                for scenario, path in results.items():
                    print(f"✅ {scenario.upper()}: {path}")
                print("="*50)
                print("Use these videos to test your AI surveillance system!")
            else:
                print("❌ Failed to generate scenarios")
                
        else:
            print(f"🚀 Generating {args.scenario} scenario...")
            if args.scenario == 'loitering':
                path = generator.generate_loitering_scenario()
            elif args.scenario == 'crowd_surge':
                path = generator.generate_crowd_surge_scenario()
            elif args.scenario == 'rapid_movement':
                path = generator.generate_rapid_movement_scenario()
            elif args.scenario == 'low_light':
                path = generator.generate_low_light_scenario()
            
            print(f"✅ {args.scenario.upper()} scenario generated: {path}")
            
    except KeyboardInterrupt:
        print("\n⏹️ Generation interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
