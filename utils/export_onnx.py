#!/usr/bin/env python3
"""
YOLO Model ONNX Export Utility
Converts PyTorch (.pt) weights to optimized ONNX (.onnx) formats.
"""

import sys
import os
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics is not installed. Run 'pip install ultralytics'")
    sys.exit(1)

def export_model(model_path: str):
    """Export YOLO model to ONNX"""
    print(f"Loading YOLO model weights from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model weights file not found at {model_path}")
        return False
        
    try:
        model = YOLO(model_path)
        print("Exporting model to ONNX format...")
        # Export to ONNX format with dynamic axis for batching and size flexibility
        exported_path = model.export(
            format="onnx",
            imgsz=640,
            dynamic=True,
            opset=12,
            simplify=True
        )
        print(f"Success! Optimized ONNX model exported to: {exported_path}")
        return True
    except Exception as e:
        print(f"Error during export: {e}")
        return False

if __name__ == "__main__":
    path = "models/sign.pt"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    
    success = export_model(path)
    sys.exit(0 if success else 1)
