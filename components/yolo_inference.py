#!/usr/bin/env python3
"""
YOLO Inference Module
Handles YOLO12 model loading and sign language detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict
# from utils.system_logger import SystemLogger


class Detection:
    """Data class for detection results"""
    def __init__(self, bbox, confidence, class_id, class_name):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.center = self._calculate_center()
    
    def _calculate_center(self):
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def get_tlwh(self):
        """Get bounding box in top-left-width-height format"""
        x1, y1, x2, y2 = self.bbox
        return [x1, y1, x2 - x1, y2 - y1]
    
    def get_area(self):
        """Calculate bounding box area"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class YOLO12Detector:
    """YOLO12 detector for sign language recognition"""
    
    def __init__(self, model_path: str = "F:/NLPPRO/sign2speech/models/sign.pt", confidence_threshold: float = 0.5):
        """
        Initialize YOLO12 detector
        
        Args:
            model_path: Path to YOLO12 model file (your custom sign.pt)
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = [
            "school", "sorry", "help", "easy", "work",
            "age", "effort", "respect", "near", "home",
            "friend", "washroom", "preset", "pass", "fail",
            "village", "eating", "drinking", "teacher", "dress",
            "message", "good"
        ]
        
        self._load_model()
    
    def _load_model(self):
        """Load YOLO12 model"""
        print(f"📦 Loading YOLO12 model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        # Verify model loaded successfully
        if not hasattr(self.model, 'names'):
            raise RuntimeError("Model does not contain class names.")
            

        print(f"✅ Model loaded successfully with {len(self.class_names)} classes")
        print(f"📋 Classes: {', '.join(self.class_names)}")

    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect sign language gestures in frame
        
        Args:
            frame: Input image frame
            
        Returns:
            List of Detection objects
        """
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                    if conf >= self.confidence_threshold:
                        # Get class name from the list
                        try:
                            class_name = self.class_names[class_id]
                        except IndexError:
                            print(f"Warning: Class ID {class_id} is out of range. Using unknown_{class_id}")
                            class_name = f"unknown_{class_id}"
                        
                        # Create detection object
                        detection = Detection(
                            bbox=box.astype(int),
                            confidence=float(conf),
                            class_id=int(class_id),
                            class_name=class_name
                        )
                        
                        detections.append(detection)
        
        return detections
    
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        return self.class_names
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        print(f"🎯 Confidence threshold updated to: {self.confidence_threshold}")
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_path': self.model_path,
            'confidence_threshold': self.confidence_threshold,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'model_loaded': self.model is not None
        }


def test_detector():
    """Test function for YOLO12 detector"""
    print("🧪 Testing YOLO12 Detector...")
    
    # Create detector (will use dummy model if real model not found)
    detector = YOLO12Detector("models/sign.pt")

    # Create test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Run detection
    detections = detector.detect(test_frame)
    
    print(f"📊 Detected {len(detections)} objects:")
    for i, det in enumerate(detections):
        result_dict = {
            "class": det.class_name,
            "confidence": det.confidence,
            "bbox": det.bbox
        }
        print(f"  {i+1}. {det.class_name} (conf: {det.confidence:.2f}, bbox: {det.bbox})")
    
    # Print model info
    info = detector.get_model_info()
    print(f"\n📋 Model Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    for det in detections:
        result_dict = {"class": det.class_name}
        print(f"Detected sign: {det.class_name}")


if __name__ == "__main__":
    test_detector()