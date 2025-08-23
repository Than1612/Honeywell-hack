import cv2
import numpy as np
from typing import Tuple, List, Optional
import os

def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize frame to target dimensions while maintaining aspect ratio"""
    height, width = frame.shape[:2]
    target_width, target_height = target_size
    
    # Calculate scaling factors
    scale_x = target_width / width
    scale_y = target_height / height
    scale = min(scale_x, scale_y)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize frame
    resized = cv2.resize(frame, (new_width, new_height))
    
    # Create canvas with target size
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Center the resized frame
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return canvas

def extract_frames(video_path: str, output_dir: str, frame_interval: int = 1) -> List[str]:
    """Extract frames from video at specified interval"""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_paths = []
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return frame_paths

def create_video_from_frames(frame_dir: str, output_path: str, fps: int = 30) -> str:
    """Create video from sequence of frames"""
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))])
    
    if not frame_files:
        raise ValueError(f"No frame files found in {frame_dir}")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)
    
    video_writer.release()
    return output_path

def apply_motion_blur(frame: np.ndarray, blur_strength: int = 5) -> np.ndarray:
    """Apply motion blur effect to frame"""
    kernel = np.ones((1, blur_strength), np.float32) / blur_strength
    blurred = cv2.filter2D(frame, -1, kernel)
    return cv2.addWeighted(frame, 0.7, blurred, 0.3, 0)

def detect_motion(frame1: np.ndarray, frame2: np.ndarray, threshold: float = 25.0) -> Tuple[bool, float]:
    """Detect motion between two consecutive frames"""
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Apply threshold
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Calculate motion percentage
    motion_pixels = np.sum(thresh > 0)
    total_pixels = thresh.size
    motion_percentage = (motion_pixels / total_pixels) * 100
    
    # Return motion detected and percentage
    return motion_percentage > 1.0, motion_percentage

def stabilize_video(frames: List[np.ndarray]) -> List[np.ndarray]:
    """Simple video stabilization using frame alignment"""
    if len(frames) < 2:
        return frames
    
    stabilized_frames = [frames[0]]
    
    for i in range(1, len(frames)):
        prev_frame = stabilized_frames[-1]
        curr_frame = frames[i]
        
        # Convert to grayscale for feature detection
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Find features
        feature_detector = cv2.SIFT_create()
        prev_kp, prev_des = feature_detector.detectAndCompute(prev_gray, None)
        curr_kp, curr_des = feature_detector.detectAndCompute(curr_gray, None)
        
        if prev_des is not None and curr_des is not None:
            # Match features
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(prev_des, curr_des, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) > 10:
                # Extract matched points
                prev_pts = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                curr_pts = np.float32([curr_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Calculate transformation matrix
                transform_matrix, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
                
                if transform_matrix is not None:
                    # Apply transformation
                    stabilized_frame = cv2.warpAffine(curr_frame, transform_matrix, (curr_frame.shape[1], curr_frame.shape[0]))
                    stabilized_frames.append(stabilized_frame)
                else:
                    stabilized_frames.append(curr_frame)
            else:
                stabilized_frames.append(curr_frame)
        else:
            stabilized_frames.append(curr_frame)
    
    return stabilized_frames

def enhance_frame(frame: np.ndarray, 
                 brightness: float = 1.0, 
                 contrast: float = 1.0, 
                 saturation: float = 1.0) -> np.ndarray:
    """Enhance frame with brightness, contrast, and saturation adjustments"""
    # Convert to HSV for saturation adjustment
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Adjust saturation
    hsv[:, :, 1] = hsv[:, :, 1] * saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Adjust brightness and contrast
    enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast, beta=(brightness - 1.0) * 255)
    
    return enhanced

def create_heatmap(motion_data: List[float], frame_shape: Tuple[int, int]) -> np.ndarray:
    """Create heatmap from motion data"""
    # Normalize motion data
    motion_array = np.array(motion_data)
    if len(motion_array) > 0:
        motion_array = (motion_array - motion_array.min()) / (motion_array.max() - motion_array.min())
    
    # Create heatmap
    heatmap = np.zeros(frame_shape[:2], dtype=np.float32)
    
    # Simple heatmap generation (in real implementation, use more sophisticated approach)
    for i, motion in enumerate(motion_array):
        if i < frame_shape[0]:
            heatmap[i, :] = motion
    
    # Convert to RGB heatmap
    heatmap_rgb = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    return heatmap_rgb
