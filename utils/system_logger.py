import logging
from ultralytics import YOLO

class SystemLogger:
    def __init__(self, name="SignLanguageToSpeechSystem"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def info(self, msg):
        self.logger.info(msg)

    def error(self, msg):
        self.logger.error(msg)

class YOLO12Detector:
    def __init__(self, model_path: str = "models/sign.pt", confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.logger = SystemLogger("YOLO12Detector")

    def load_model(self):
        """Load the YOLOv8 model."""
        try:
            self.model = YOLO(self.model_path)
            self.logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise

    def predict(self, image):
        """Run inference on the given image."""
        if self.model is None:
            self.logger.error("Model not loaded. Call 'load_model' before prediction.")
            raise RuntimeError("Model not loaded. Call 'load_model' before prediction.")

        try:
            results = self.model(image, conf=self.confidence_threshold)
            self.logger.info(f"Prediction made with confidence threshold {self.confidence_threshold}")
            return results
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

detector = YOLO("models/sign.pt")