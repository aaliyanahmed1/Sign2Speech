#!/usr/bin/env python3
"""
Video Utilities Module
OpenCV helpers and video processing utilities for the sign language system
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import time
from pathlib import Path
import threading
from collections import deque


class VideoProcessor:
    """Video processing utilities and helpers"""
    
    def __init__(self):
        """
        Initialize video processor
        """
        self.frame_buffer = deque(maxlen=30)  # Keep last 30 frames
        self.fps_calculator = FPSCalculator()
        
        # Video recording settings
        self.is_recording = False
        self.video_writer = None
        self.recording_start_time = None
        
        print("📹 Video processor initialized")
    
    def preprocess_frame(self, frame: np.ndarray, 
                        target_size: Optional[Tuple[int, int]] = None,
                        normalize: bool = False,
                        enhance_contrast: bool = False) -> np.ndarray:
        """Preprocess frame for better detection"""
        processed_frame = frame.copy()
        
        # Resize if target size specified
        if target_size:
            processed_frame = cv2.resize(processed_frame, target_size)
        
        # Enhance contrast if requested
        if enhance_contrast:
            processed_frame = self.enhance_contrast(processed_frame)
        
        # Normalize if requested
        if normalize:
            processed_frame = processed_frame.astype(np.float32) / 255.0
        
        return processed_frame
    
    def enhance_contrast(self, frame: np.ndarray, 
                        clip_limit: float = 2.0,
                        tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Enhance frame contrast using CLAHE"""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def stabilize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Basic frame stabilization using previous frames"""
        if len(self.frame_buffer) < 2:
            self.frame_buffer.append(frame)
            return frame
        
        # Simple stabilization by averaging with previous frame
        prev_frame = self.frame_buffer[-1]
        stabilized = cv2.addWeighted(frame, 0.7, prev_frame, 0.3, 0)
        
        self.frame_buffer.append(frame)
        return stabilized
    
    def detect_motion(self, frame: np.ndarray, 
                     threshold: float = 25.0) -> Tuple[bool, np.ndarray]:
        """Detect motion in the frame"""
        if len(self.frame_buffer) < 2:
            self.frame_buffer.append(frame)
            return False, np.zeros_like(frame[:, :, 0])
        
        # Convert frames to grayscale
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(self.frame_buffer[-1], cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        diff = cv2.absdiff(gray_current, gray_prev)
        
        # Apply threshold
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Calculate motion percentage
        motion_pixels = np.sum(thresh > 0)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        motion_percentage = (motion_pixels / total_pixels) * 100
        
        has_motion = motion_percentage > 1.0  # 1% threshold
        
        self.frame_buffer.append(frame)
        return has_motion, thresh
    
    def crop_roi(self, frame: np.ndarray, 
                roi: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop region of interest from frame"""
        x, y, w, h = roi
        return frame[y:y+h, x:x+w]
    
    def resize_frame(self, frame: np.ndarray, 
                    target_size: Tuple[int, int],
                    maintain_aspect: bool = True) -> np.ndarray:
        """Resize frame with optional aspect ratio maintenance"""
        if not maintain_aspect:
            return cv2.resize(frame, target_size)
        
        h, w = frame.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Create canvas and center the resized frame
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def apply_blur(self, frame: np.ndarray, 
                  blur_type: str = 'gaussian',
                  kernel_size: int = 5) -> np.ndarray:
        """Apply blur to frame"""
        if blur_type == 'gaussian':
            return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        elif blur_type == 'median':
            return cv2.medianBlur(frame, kernel_size)
        elif blur_type == 'bilateral':
            return cv2.bilateralFilter(frame, kernel_size, 75, 75)
        else:
            return frame
    
    def adjust_brightness_contrast(self, frame: np.ndarray,
                                 brightness: int = 0,
                                 contrast: float = 1.0) -> np.ndarray:
        """Adjust brightness and contrast"""
        adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        return adjusted
    
    def start_recording(self, output_path: str, 
                       fps: float = 30.0,
                       frame_size: Tuple[int, int] = (640, 480),
                       codec: str = 'XVID') -> bool:
        """Start video recording"""
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
            
            if self.video_writer.isOpened():
                self.is_recording = True
                self.recording_start_time = time.time()
                print(f"🔴 Recording started: {output_path}")
                return True
            else:
                print(f"❌ Failed to start recording: {output_path}")
                return False
                
        except Exception as e:
            print(f"❌ Recording error: {e}")
            return False
    
    def record_frame(self, frame: np.ndarray) -> bool:
        """Record a frame to video file"""
        if not self.is_recording or self.video_writer is None:
            return False
        
        try:
            self.video_writer.write(frame)
            return True
        except Exception as e:
            print(f"❌ Frame recording error: {e}")
            return False
    
    def stop_recording(self) -> Optional[float]:
        """Stop video recording and return duration"""
        if not self.is_recording:
            return None
        
        try:
            duration = None
            if self.recording_start_time:
                duration = time.time() - self.recording_start_time
            
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            self.is_recording = False
            self.recording_start_time = None
            
            print(f"⏹️ Recording stopped. Duration: {duration:.2f}s" if duration else "⏹️ Recording stopped")
            return duration
            
        except Exception as e:
            print(f"❌ Stop recording error: {e}")
            return None
    
    def get_frame_info(self, frame: np.ndarray) -> Dict[str, Any]:
        """Get information about a frame"""
        h, w = frame.shape[:2]
        channels = frame.shape[2] if len(frame.shape) > 2 else 1
        
        return {
            'width': w,
            'height': h,
            'channels': channels,
            'dtype': str(frame.dtype),
            'size_bytes': frame.nbytes,
            'aspect_ratio': round(w / h, 2)
        }
    
    def calculate_fps(self) -> float:
        """Calculate current FPS"""
        return self.fps_calculator.get_fps()
    
    def update_fps(self):
        """Update FPS calculation"""
        self.fps_calculator.update()


class FPSCalculator:
    """Calculate FPS for video processing"""
    
    def __init__(self, buffer_size: int = 30):
        """
        Initialize FPS calculator
        
        Args:
            buffer_size: Number of frames to average over
        """
        self.buffer_size = buffer_size
        self.frame_times = deque(maxlen=buffer_size)
        self.last_time = time.time()
    
    def update(self):
        """Update with new frame"""
        current_time = time.time()
        self.frame_times.append(current_time)
        self.last_time = current_time
    
    def get_fps(self) -> float:
        """Get current FPS"""
        if len(self.frame_times) < 2:
            return 0.0
        
        time_diff = self.frame_times[-1] - self.frame_times[0]
        if time_diff == 0:
            return 0.0
        
        fps = (len(self.frame_times) - 1) / time_diff
        return fps


class CameraManager:
    """Manage camera/video source"""
    
    def __init__(self, source: int = 0):
        """
        Initialize camera manager
        
        Args:
            source: Camera index or video file path
        """
        self.source = source
        self.cap = None
        self.is_opened = False
        self.frame_count = 0
        
        # Camera properties
        self.width = 640
        self.height = 480
        self.fps = 30
    
    def open(self) -> bool:
        """Open camera/video source"""
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                print(f"❌ Failed to open video source: {self.source}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Get actual properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.is_opened = True
            print(f"📹 Video source opened: {self.width}x{self.height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            print(f"❌ Camera open error: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from camera"""
        if not self.is_opened or self.cap is None:
            return False, None
        
        try:
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
            return ret, frame
            
        except Exception as e:
            print(f"❌ Frame read error: {e}")
            return False, None
    
    def set_resolution(self, width: int, height: int) -> bool:
        """Set camera resolution"""
        if not self.is_opened:
            self.width = width
            self.height = height
            return True
        
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.width = actual_width
            self.height = actual_height
            
            print(f"📐 Resolution set to: {self.width}x{self.height}")
            return True
            
        except Exception as e:
            print(f"❌ Resolution setting error: {e}")
            return False
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information"""
        info = {
            'source': self.source,
            'is_opened': self.is_opened,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'frame_count': self.frame_count
        }
        
        if self.is_opened and self.cap:
            try:
                info.update({
                    'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                    'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
                    'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
                    'hue': self.cap.get(cv2.CAP_PROP_HUE),
                    'gain': self.cap.get(cv2.CAP_PROP_GAIN),
                    'exposure': self.cap.get(cv2.CAP_PROP_EXPOSURE)
                })
            except:
                pass  # Some properties might not be available
        
        return info
    
    def close(self):
        """Close camera/video source"""
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.is_opened = False
        print("📹 Video source closed")


def test_video_utils():
    """Test function for video utilities"""
    print("🧪 Testing Video Utils...")
    
    # Test video processor
    processor = VideoProcessor()
    
    # Create test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("\n📹 Testing video processing:")
    
    # Test preprocessing
    processed = processor.preprocess_frame(test_frame, target_size=(320, 240))
    print(f"  Preprocessed frame shape: {processed.shape}")
    
    # Test contrast enhancement
    enhanced = processor.enhance_contrast(test_frame)
    print(f"  Enhanced frame shape: {enhanced.shape}")
    
    # Test motion detection
    has_motion, motion_mask = processor.detect_motion(test_frame)
    print(f"  Motion detected: {has_motion}")
    
    # Test frame info
    info = processor.get_frame_info(test_frame)
    print(f"  Frame info: {info}")
    
    # Test FPS calculator
    fps_calc = FPSCalculator()
    for _ in range(10):
        fps_calc.update()
        time.sleep(0.01)  # Simulate frame processing time
    
    fps = fps_calc.get_fps()
    print(f"  Calculated FPS: {fps:.2f}")
    
    # Test camera manager (without actually opening camera)
    camera = CameraManager(0)
    camera_info = camera.get_camera_info()
    print(f"\n📷 Camera info: {camera_info}")
    
    print("✅ Video utilities test completed")


if __name__ == "__main__":
    test_video_utils()