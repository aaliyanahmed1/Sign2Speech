#!/usr/bin/env python3
"""
Drawing Utilities Module
Handles visualization of bounding boxes, track IDs, recognized words, and UI elements
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import colorsys
import math


class DrawingUtils:
    """Utilities for drawing annotations on video frames"""
    
    def __init__(self):
        """
        Initialize drawing utilities with color schemes and fonts
        """
        # Color schemes
        self.colors = self._generate_colors(50)  # Generate 50 distinct colors
        self.confidence_colors = {
            'high': (0, 255, 0),      # Green for high confidence
            'medium': (0, 255, 255),   # Yellow for medium confidence
            'low': (0, 0, 255)         # Red for low confidence
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        self.text_color = (255, 255, 255)  # White text
        self.text_bg_color = (0, 0, 0)     # Black background
        
        # UI settings
        self.box_thickness = 2
        self.text_padding = 5
        self.ui_alpha = 0.7  # Transparency for UI elements
        
        print("🎨 Drawing utilities initialized")
    
    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for different tracks"""
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            saturation = 0.8
            value = 0.9
            
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            # Convert to BGR for OpenCV
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        
        return colors
    
    def get_confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        """Get color based on confidence level"""
        if confidence >= 0.8:
            return self.confidence_colors['high']
        elif confidence >= 0.6:
            return self.confidence_colors['medium']
        else:
            return self.confidence_colors['low']
    
    def get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get consistent color for a track ID"""
        return self.colors[track_id % len(self.colors)]
    
    def draw_bounding_box(self, frame: np.ndarray, bbox: List[int], 
                         color: Tuple[int, int, int], 
                         thickness: Optional[int] = None) -> np.ndarray:
        """Draw a bounding box on the frame"""
        if thickness is None:
            thickness = self.box_thickness
        
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        return frame
    
    def draw_text_with_background(self, frame: np.ndarray, text: str, 
                                position: Tuple[int, int], 
                                font_scale: Optional[float] = None,
                                text_color: Optional[Tuple[int, int, int]] = None,
                                bg_color: Optional[Tuple[int, int, int]] = None,
                                padding: Optional[int] = None) -> np.ndarray:
        """Draw text with background rectangle"""
        if font_scale is None:
            font_scale = self.font_scale
        if text_color is None:
            text_color = self.text_color
        if bg_color is None:
            bg_color = self.text_bg_color
        if padding is None:
            padding = self.text_padding
        
        x, y = position
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, font_scale, self.font_thickness
        )
        
        # Draw background rectangle
        bg_x1 = x - padding
        bg_y1 = y - text_height - padding
        bg_x2 = x + text_width + padding
        bg_y2 = y + baseline + padding
        
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
        
        # Draw text
        cv2.putText(frame, text, (x, y), self.font, font_scale, 
                   text_color, self.font_thickness)
        
        return frame
    
    def draw_text(self, frame: np.ndarray, text: str, 
                 position: Tuple[int, int],
                 font_scale: Optional[float] = None,
                 color: Optional[Tuple[int, int, int]] = None,
                 thickness: Optional[int] = None) -> np.ndarray:
        """Draw simple text without background"""
        if font_scale is None:
            font_scale = self.font_scale
        if color is None:
            color = self.text_color
        if thickness is None:
            thickness = self.font_thickness
        
        cv2.putText(frame, text, position, self.font, font_scale, 
                   color, thickness)
        return frame
    
    def draw_detection(self, frame: np.ndarray, detection, 
                      show_confidence: bool = True) -> np.ndarray:
        """Draw a single detection with bounding box and label"""
        # Get color based on confidence
        color = self.get_confidence_color(detection.confidence)
        
        # Draw bounding box
        frame = self.draw_bounding_box(frame, detection.bbox, color)
        
        # Prepare label text
        label = detection.class_name
        if show_confidence:
            label += f" {detection.confidence:.2f}"
        
        # Draw label
        x1, y1, x2, y2 = detection.bbox
        label_position = (x1, y1 - 10 if y1 > 30 else y1 + 30)
        frame = self.draw_text_with_background(frame, label, label_position)
        
        return frame
    
    def draw_track(self, frame: np.ndarray, track, 
                  show_confidence: bool = True,
                  show_track_id: bool = True) -> np.ndarray:
        """Draw a single track with bounding box, label, and track ID"""
        # Get color for this track
        color = self.get_track_color(track.track_id)
        
        # Draw bounding box
        frame = self.draw_bounding_box(frame, track.bbox, color)
        
        # Prepare label text
        label_parts = []
        
        if show_track_id:
            label_parts.append(f"ID:{track.track_id}")
        
        label_parts.append(track.class_name)
        
        if show_confidence:
            label_parts.append(f"{track.confidence:.2f}")
        
        label = " ".join(label_parts)
        
        # Draw label
        x1, y1, x2, y2 = track.bbox
        label_position = (x1, y1 - 10 if y1 > 30 else y1 + 30)
        frame = self.draw_text_with_background(frame, label, label_position, 
                                             bg_color=color)
        
        # Draw center point
        center_x, center_y = track.center
        cv2.circle(frame, (center_x, center_y), 3, color, -1)
        
        return frame
    
    def draw_tracks(self, frame: np.ndarray, tracks: List, 
                   show_confidence: bool = True,
                   show_track_id: bool = True) -> np.ndarray:
        """Draw multiple tracks on the frame"""
        for track in tracks:
            frame = self.draw_track(frame, track, show_confidence, show_track_id)
        
        return frame
    
    def draw_detections(self, frame: np.ndarray, detections: List,
                       show_confidence: bool = True) -> np.ndarray:
        """Draw multiple detections on the frame"""
        for detection in detections:
            frame = self.draw_detection(frame, detection, show_confidence)
        
        return frame
    
    def draw_info_panel(self, frame: np.ndarray, info: Dict[str, Any],
                       position: Tuple[int, int] = (10, 10),
                       panel_width: int = 300) -> np.ndarray:
        """Draw an information panel with system stats"""
        x, y = position
        line_height = 25
        current_y = y
        
        # Draw panel background
        panel_height = len(info) * line_height + 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - 10), 
                     (x + panel_width, y + panel_height), 
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 1 - self.ui_alpha, overlay, self.ui_alpha, 0)
        
        # Draw info text
        for key, value in info.items():
            text = f"{key}: {value}"
            frame = self.draw_text(frame, text, (x, current_y + 15), 
                                 font_scale=0.5, color=(255, 255, 255))
            current_y += line_height
        
        return frame
    
    def draw_sentence_display(self, frame: np.ndarray, 
                            current_sentence: str,
                            recent_sentences: List[str] = None,
                            position: str = 'bottom') -> np.ndarray:
        """Draw current and recent sentences on the frame"""
        h, w = frame.shape[:2]
        
        if position == 'bottom':
            base_y = h - 80
        else:  # top
            base_y = 50
        
        # Draw current sentence (larger, highlighted)
        if current_sentence:
            # Background for current sentence
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, base_y - 30), (w - 10, base_y + 10), 
                         (0, 50, 0), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Current sentence text
            frame = self.draw_text(frame, f"Current: {current_sentence}", 
                                 (20, base_y), font_scale=0.7, 
                                 color=(0, 255, 0), thickness=2)
        
        # Draw recent sentences
        if recent_sentences:
            for i, sentence in enumerate(recent_sentences[-3:]):  # Show last 3
                y_pos = base_y + 30 + (i * 20)
                if y_pos < h - 10:  # Make sure it fits on screen
                    frame = self.draw_text(frame, f"  {sentence}", 
                                         (20, y_pos), font_scale=0.5, 
                                         color=(200, 200, 200))
        
        return frame
    
    def draw_progress_bar(self, frame: np.ndarray, progress: float,
                         position: Tuple[int, int] = (10, 10),
                         size: Tuple[int, int] = (200, 20),
                         color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw a progress bar"""
        x, y = position
        width, height = size
        
        # Clamp progress between 0 and 1
        progress = max(0, min(1, progress))
        
        # Draw background
        cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), -1)
        
        # Draw progress
        progress_width = int(width * progress)
        if progress_width > 0:
            cv2.rectangle(frame, (x, y), (x + progress_width, y + height), color, -1)
        
        # Draw border
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 1)
        
        # Draw percentage text
        text = f"{int(progress * 100)}%"
        text_x = x + width // 2 - 15
        text_y = y + height // 2 + 5
        frame = self.draw_text(frame, text, (text_x, text_y), 
                             font_scale=0.4, color=(255, 255, 255))
        
        return frame
    
    def draw_fps_counter(self, frame: np.ndarray, fps: float,
                        position: Tuple[int, int] = None) -> np.ndarray:
        """Draw FPS counter"""
        if position is None:
            h, w = frame.shape[:2]
            position = (w - 100, 30)
        
        fps_text = f"FPS: {fps:.1f}"
        
        # Color based on FPS
        if fps >= 25:
            color = (0, 255, 0)  # Green
        elif fps >= 15:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        frame = self.draw_text_with_background(frame, fps_text, position, 
                                             text_color=color)
        return frame
    
    def draw_crosshair(self, frame: np.ndarray, 
                      center: Optional[Tuple[int, int]] = None,
                      size: int = 20,
                      color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """Draw crosshair at center or specified position"""
        h, w = frame.shape[:2]
        
        if center is None:
            center = (w // 2, h // 2)
        
        cx, cy = center
        
        # Draw crosshair lines
        cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 1)
        cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 1)
        
        # Draw center dot
        cv2.circle(frame, (cx, cy), 2, color, -1)
        
        return frame
    
    def draw_grid(self, frame: np.ndarray, grid_size: int = 50,
                 color: Tuple[int, int, int] = (50, 50, 50)) -> np.ndarray:
        """Draw grid overlay for debugging"""
        h, w = frame.shape[:2]
        
        # Vertical lines
        for x in range(0, w, grid_size):
            cv2.line(frame, (x, 0), (x, h), color, 1)
        
        # Horizontal lines
        for y in range(0, h, grid_size):
            cv2.line(frame, (0, y), (w, y), color, 1)
        
        return frame
    
    def create_legend(self, frame: np.ndarray, 
                     class_names: List[str],
                     position: Tuple[int, int] = (10, 10)) -> np.ndarray:
        """Create a legend showing class names and colors"""
        x, y = position
        line_height = 25
        
        # Draw legend background
        legend_width = 200
        legend_height = len(class_names) * line_height + 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - 10), 
                     (x + legend_width, y + legend_height), 
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        # Draw legend items
        for i, class_name in enumerate(class_names):
            item_y = y + (i * line_height) + 15
            
            # Draw color box
            color = self.colors[i % len(self.colors)]
            cv2.rectangle(frame, (x, item_y - 10), (x + 15, item_y + 5), color, -1)
            
            # Draw class name
            frame = self.draw_text(frame, class_name, (x + 25, item_y), 
                                 font_scale=0.5, color=(255, 255, 255))
        
        return frame


def test_drawing_utils():
    """Test function for drawing utilities"""
    print("🧪 Testing Drawing Utils...")
    
    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    drawer = DrawingUtils()
    
    # Test bounding box
    frame = drawer.draw_bounding_box(frame, [100, 100, 200, 200], (0, 255, 0))
    
    # Test text with background
    frame = drawer.draw_text_with_background(frame, "Test Text", (100, 80))
    
    # Test info panel
    info = {
        'FPS': 30.5,
        'Tracks': 3,
        'Detections': 5,
        'Status': 'Running'
    }
    frame = drawer.draw_info_panel(frame, info)
    
    # Test progress bar
    frame = drawer.draw_progress_bar(frame, 0.75, (100, 250))
    
    # Test FPS counter
    frame = drawer.draw_fps_counter(frame, 28.3)
    
    # Test crosshair
    frame = drawer.draw_crosshair(frame)
    
    # Test sentence display
    frame = drawer.draw_sentence_display(
        frame, 
        "I love you", 
        ["Hello", "Thank you", "How are you"]
    )
    
    print("✅ Drawing utilities test completed")
    print("💡 To see the visual result, save the frame and view it")
    
    # Save test frame
    cv2.imwrite("test_drawing_output.jpg", frame)
    print("📁 Test output saved as 'test_drawing_output.jpg'")


if __name__ == "__main__":
    test_drawing_utils()