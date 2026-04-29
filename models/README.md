# Models Directory

This directory contains the trained models used by the Sign Language to Speech System.

## Required Models

### 1. YOLO12 Sign Language Model
- **File**: `yolov8_sign.pt`
- **Purpose**: Detects sign language gestures in video frames
- **Format**: PyTorch model file (.pt)
- **Size**: Typically 6-50MB depending on model variant

### 2. DeepSORT Checkpoint
- **File**: `deepsort/ckpt.t7`
- **Purpose**: Feature extraction for object tracking
- **Format**: Torch checkpoint (.t7)
- **Size**: ~1-5MB

## Model Setup

### Option 1: Use Dummy Detector (Testing)
If no model is present, the system will automatically use a dummy detector that generates random detections for testing purposes.

### Option 2: Download Pre-trained Model
```bash
# Download a general YOLOv8 model (you'll need to retrain for sign language)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
mv yolov8n.pt yolov8_sign.pt
```

### Option 3: Train Your Own Model

1. **Prepare Dataset**:
   - Collect sign language images/videos
   - Annotate with bounding boxes and class labels
   - Format in YOLO format (txt files with normalized coordinates)

2. **Training Script**:
```python
from ultralytics import YOLO

# Load a pre-trained model
model = YOLO('yolo12n.pt')  # nano, small, medium, large, xlarge

# Train the model
results = model.train(
    data='../data/sign_dataset/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='sign_language_model'
)

# Save the trained model
model.save('yolo12_sign.pt')
```

3. **Dataset YAML Example** (`../data/sign_dataset/dataset.yaml`):
```yaml
path: ../data/sign_dataset
train: images/train
val: images/val
test: images/test

names:
  0: hello
  1: goodbye
  2: I
  3: love
  4: you
  5: thank_you
  6: please
  7: yes
  8: no
  # Add more sign classes as needed
```

## DeepSORT Setup

### Download DeepSORT Model
```bash
# Create deepsort directory
mkdir -p deepsort

# Download the checkpoint (example URL - replace with actual)
wget https://drive.google.com/uc?id=1_qwTWdzT9dWNudpusgKavj_4elGgbkUN -O deepsort/ckpt.t7
```

### Alternative: Use Simple Tracker
If DeepSORT is not available, the system will fall back to a simple tracker that maintains basic object tracking without deep features.

## Model Performance Guidelines

#

## Troubleshooting

### Model Loading Issues
```python
# Test model loading
from ultralytics import YOLO
try:
    model = YOLO('models/yolo12_sign.pt')
    print("Model loaded successfully")
    print(f"Model classes: {model.names}")
except Exception as e:
    print(f"Error loading model: {e}")
```

### Common Problems
1. **File not found**: Ensure model path is correct
2. **Corrupted model**: Re-download or retrain
3. **Version mismatch**: Update ultralytics package
4. **CUDA issues**: Check PyTorch CUDA compatibility

## Model Optimization

### Export to Different Formats
```python
from ultralytics import YOLO

model = YOLO('yolo12_sign.pt')

# Export to ONNX for faster inference
model.export(format='onnx')

# Export to TensorRT (NVIDIA GPUs)
model.export(format='engine')

# Export to CoreML (Apple devices)
model.export(format='coreml')
```

### Quantization for Mobile
```python
# Export with quantization for mobile deployment
model.export(format='onnx', int8=True)
```

---

**Note**: This directory should contain your trained models. The system will work with dummy data if no models are present, making it easy to test the pipeline before training custom models.
