from typing import Optional, List, Dict
from pydantic import BaseModel, Field

class SpeakRequest(BaseModel):
    sentence: str = Field(..., min_length=1, max_length=2000)
    voice_id: Optional[str] = "ur-PK-AsmaNeural"


class SpeakResponse(BaseModel):
    audio_url: str
    sentence: str


class UploadResponse(BaseModel):
    job_id: str
    gestures: List[str]
    sentence: str
    confidence_scores: List[float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    version: str
    class_count: int


class AnalyticsResponse(BaseModel):
    total_sessions: int
    gestures_detected: int
    avg_confidence: float
    sentences_spoken: int
    gesture_frequency: Dict[str, int]
    session_history: List[dict]
    recent_detections: List[dict]


class TranslateRequest(BaseModel):
    sentence: str
    use_ollama: bool = True
    voice_id: Optional[str] = None
