import os

BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
API_KEY = os.getenv("API_KEY", "")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
MODEL_PATH = os.getenv("MODEL_PATH", "models/sign.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
LOG_DIR = os.getenv("LOG_DIR", "logs")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))

CLASS_NAMES = [
    "school", "sorry", "help", "easy", "work",
    "age", "effort", "respect", "near", "home",
    "friend", "washroom", "preset", "pass", "fail",
    "village", "eating", "drinking", "teacher", "dress",
    "message", "good",
]

CLASS_CATEGORIES = {
    "basic": ["school", "sorry", "help", "easy", "work", "age", "effort", "respect"],
    "location": ["near", "home", "village", "washroom"],
    "social": ["friend", "teacher", "message", "good"],
    "actions": ["eating", "drinking", "pass", "fail"],
    "settings": ["preset", "dress"],
}
