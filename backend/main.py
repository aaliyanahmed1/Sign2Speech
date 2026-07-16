import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

logger = logging.getLogger("sign2speech.backend")

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
from backend.config import (
    BACKEND_HOST, BACKEND_PORT, CORS_ORIGINS, API_KEY, MAX_UPLOAD_MB,
    MODEL_PATH, CONFIDENCE_THRESHOLD, LOG_DIR, OLLAMA_URL, OLLAMA_MODEL,
    OLLAMA_TIMEOUT, CLASS_NAMES, CLASS_CATEGORIES
)

# ---------------------------------------------------------------------------
# Startup tracking
# ---------------------------------------------------------------------------
APP_START = time.time()
VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Sign2Speech API", version=VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load YOLO model
# ---------------------------------------------------------------------------
model: Optional[YOLO] = None
try:
    resolved_model_path = MODEL_PATH
    if not os.path.exists(resolved_model_path):
        onnx_fallback = "models/sign.onnx"
        pt_fallback = "models/sign.pt"
        if os.path.exists(onnx_fallback):
            resolved_model_path = onnx_fallback
        elif os.path.exists(pt_fallback):
            resolved_model_path = pt_fallback

    if os.path.exists(resolved_model_path):
        model = YOLO(resolved_model_path)
        logger.info("YOLO model loaded from %s", resolved_model_path)
    else:
        logger.warning("Model file not found (tried %s, models/sign.onnx, models/sign.pt) — upload endpoint will return errors", MODEL_PATH)
except Exception as exc:
    logger.error("Failed to load YOLO model: %s", exc)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------
async def verify_api_key(authorization: Optional[str] = Header(None)):
    if not API_KEY:
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------
from backend.schemas import (
    SpeakRequest, SpeakResponse, UploadResponse, HealthResponse,
    AnalyticsResponse, TranslateRequest
)


# ---------------------------------------------------------------------------
# In-memory session / analytics store
# ---------------------------------------------------------------------------
session_store: dict[str, list] = {}
analytics_data = {
    "total_sessions": 1,
    "gestures_detected": 0,
    "total_confidence": 0.0,
    "sentences_spoken": 0,
    "gesture_frequency": {name: 0 for name in CLASS_NAMES},
    "session_history": [],
    "recent_detections": [],
}


def _update_analytics(gestures: list[str], confidences: list[float], sentence: str) -> None:
    analytics_data["gestures_detected"] += len(gestures)
    for g in gestures:
        analytics_data["gesture_frequency"][g] = analytics_data["gesture_frequency"].get(g, 0) + 1
    if confidences:
        analytics_data["total_confidence"] += sum(confidences)
    if sentence:
        analytics_data["sentences_spoken"] += 1
    for g, c in zip(gestures, confidences):
        analytics_data["recent_detections"].append({
            "gesture": g,
            "confidence": round(c, 3),
            "timestamp": datetime.now().isoformat(),
        })
        if len(analytics_data["recent_detections"]) > 200:
            analytics_data["recent_detections"] = analytics_data["recent_detections"][-200:]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["root"])
def root():
    return {"status": "Sign2Speech API is running", "version": VERSION}


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        uptime_seconds=round(time.time() - APP_START, 2),
        version=VERSION,
        class_count=len(CLASS_NAMES),
    )


@app.get("/api/classes")
def get_classes():
    classes = []
    for idx, name in enumerate(CLASS_NAMES):
        category = "unknown"
        for cat, names in CLASS_CATEGORIES.items():
            if name in names:
                category = cat
                break
        classes.append({"id": idx, "name": name, "category": category})
    return {"classes": classes}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB limit")

    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    gestures: list[str] = []
    confidences: list[float] = []

    if model is not None:
        import re
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        for r in results:
            if r.boxes is not None:
                for cls_id_raw, conf_raw in zip(
                    r.boxes.cls.tolist(), r.boxes.conf.tolist()
                ):
                    raw_name = model.names[int(cls_id_raw)]
                    numeric_match = re.search(r"\d+", raw_name)
                    if numeric_match:
                        idx = int(numeric_match.group()) - 1
                        class_name = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else raw_name
                    else:
                        class_name = raw_name
                    gestures.append(class_name)
                    confidences.append(float(conf_raw))

        gestures = list(dict.fromkeys(gestures))
        confidences = confidences[: len(gestures)]

    sentence = " ".join(gestures).title() if gestures else "No gesture detected"

    job_id = f"job_{int(time.time() * 1000)}"
    session_store[job_id] = gestures

    _update_analytics(gestures, confidences, sentence)

    return UploadResponse(
        job_id=job_id,
        gestures=gestures,
        sentence=sentence,
        confidence_scores=confidences,
    )


@app.get("/api/result/{job_id}")
def get_result(job_id: str):
    gestures = session_store.get(job_id)
    if gestures is None:
        raise HTTPException(status_code=404, detail="Job not found")
    sentence = " ".join(gestures).title()
    return {"job_id": job_id, "gestures": gestures, "sentence": sentence}


@app.post("/api/speak", response_model=SpeakResponse)
async def speak(req: SpeakRequest):
    from components.tts_engine import TTSEngine

    tts = TTSEngine(
        voice_id=req.voice_id,
        output_dir=os.getenv("TTS_OUTPUT_DIR", "logs/audio_outputs")
    )
    audio_path = tts.synthesize_speech(req.sentence, play_audio=False)

    if audio_path and os.path.exists(audio_path):
        return SpeakResponse(
            audio_url=f"/api/audio/{os.path.basename(audio_path)}",
            sentence=req.sentence,
        )

    return SpeakResponse(audio_url="", sentence=req.sentence)


@app.get("/api/audio/{filename}")
def serve_audio(filename: str):
    audio_dir = Path(os.getenv("TTS_OUTPUT_DIR", "logs/audio_outputs"))
    file_path = audio_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    media_type = "audio/wav"
    if filename.endswith(".mp3"):
        media_type = "audio/mpeg"
        
    return FileResponse(str(file_path), media_type=media_type)


@app.get("/api/analytics", response_model=AnalyticsResponse)
def get_analytics():
    avg_conf = 0.0
    if analytics_data["gestures_detected"] > 0:
        avg_conf = analytics_data["total_confidence"] / analytics_data["gestures_detected"]
    return AnalyticsResponse(
        total_sessions=analytics_data["total_sessions"],
        gestures_detected=analytics_data["gestures_detected"],
        avg_confidence=round(avg_conf, 3),
        sentences_spoken=analytics_data["sentences_spoken"],
        gesture_frequency=analytics_data["gesture_frequency"],
        session_history=analytics_data["session_history"],
        recent_detections=analytics_data["recent_detections"][-50:],
    )


@app.get("/api/session-history")
def get_session_history():
    log_path = Path(LOG_DIR) / "session_data.json"
    if log_path.exists():
        import json
        with open(log_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"session_id": "none", "sentences": [], "statistics": {}}


# ---------------------------------------------------------------------------
# SignSpeak session management
# ---------------------------------------------------------------------------
signspeak_store: list[dict] = []

class SignSpeakEntry(BaseModel):
    sentence: str = Field(..., min_length=1, max_length=2000)
    signs: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

class SignSpeakDeleteResponse(BaseModel):
    status: str
    id: str


@app.post("/api/session/save")
async def save_signspeak_entry(req: SignSpeakEntry):
    entry = {
        "id": f"ss_{int(time.time() * 1000)}",
        "sentence": req.sentence,
        "signs": req.signs,
        "confidence": req.confidence,
        "timestamp": datetime.now().isoformat(),
        "pinned": False,
    }
    signspeak_store.insert(0, entry)
    # Keep max 200 entries in memory
    while len(signspeak_store) > 200:
        signspeak_store.pop()
    return {"status": "ok", "entry": entry}


@app.delete("/api/session/{entry_id}")
async def delete_signspeak_entry(entry_id: str):
    global signspeak_store
    signspeak_store = [e for e in signspeak_store if e.get("id") != entry_id]
    return SignSpeakDeleteResponse(status="deleted", id=entry_id)


@app.delete("/api/session/clear")
async def clear_all_signspeak():
    global signspeak_store
    signspeak_store = []
    return {"status": "cleared"}


@app.get("/api/signspeak/history")
async def get_signspeak_history():
    return {"entries": signspeak_store}


@app.post("/api/calibrate")
async def upload_calibration(file: UploadFile = File(...), side: str = "left"):
    if side not in ("left", "right"):
        raise HTTPException(status_code=400, detail="side must be 'left' or 'right'")
    contents = await file.read()
    calib_dir = Path(LOG_DIR) / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)
    path = calib_dir / f"hand_{side}_{int(time.time())}.jpg"
    with open(path, "wb") as f:
        f.write(contents)
    return {"status": "calibrated", "side": side, "file": str(path)}





@app.post("/api/translate")
async def translate_sentence(req: TranslateRequest):
    if not req.sentence:
        return {"translated": ""}
    
    if req.use_ollama:
        try:
            import requests
            words = [w.strip() for w in req.sentence.split() if w.strip()]
            if not words:
                return {"translated": ""}
            
            is_urdu = req.voice_id and req.voice_id.startswith("ur-")
            if is_urdu:
                prompt = (
                    f"Translate and combine the following sign language gestures in order into a single, "
                    f"grammatically correct and natural Urdu sentence written in Urdu script (e.g., 'آپ کیسے ہیں'): {', '.join(words)}. "
                    f"Only return the final Urdu sentence in Urdu script, no English translation and no other text."
                )
            else:
                if len(words) == 1:
                    from components.yolo2voice_pipeline import generate_sentence_with_ollama
                    sentence = generate_sentence_with_ollama(words[0])
                    return {"translated": sentence}
                prompt = (
                    f"Combine the following sign language gestures in order into a single, "
                    f"grammatically correct and natural English sentence: {', '.join(words)}. "
                    f"Only return the final sentence, no other text."
                )
                
            payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
            response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            response.raise_for_status()
            sentence = response.json().get("response", "").strip()
            if (sentence.startswith('"') and sentence.endswith('"')) or (sentence.startswith("'") and sentence.endswith("'")):
                sentence = sentence[1:-1]
            return {"translated": sentence}
        except Exception as e:
            logger.warning("Ollama translation failed, falling back: %s", e)
            
    return {"translated": req.sentence.title()}


# ---------------------------------------------------------------------------
# WebSocket — real-time detection stream
# ---------------------------------------------------------------------------
@app.websocket("/api/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")

    import json
    import base64
    import re

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue
                
                # Check for frame JSON message
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "frame" and "image" in msg:
                        image_data = msg["image"]
                        if "," in image_data:
                            header, encoded = image_data.split(",", 1)
                        else:
                            encoded = image_data
                        
                        data_bytes = base64.b64decode(encoded)
                        nparr = np.frombuffer(data_bytes, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None and model is not None:
                            results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
                            detected_gesture = None
                            max_conf = 0.0
                            
                            for r in results:
                                if r.boxes is not None:
                                    for cls_id_raw, conf_raw in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                                        raw_name = model.names[int(cls_id_raw)]
                                        numeric_match = re.search(r"\d+", raw_name)
                                        if numeric_match:
                                            idx = int(numeric_match.group()) - 1
                                            class_name = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else raw_name
                                        else:
                                            class_name = raw_name
                                        
                                        if conf_raw > max_conf:
                                            max_conf = float(conf_raw)
                                            detected_gesture = class_name
                            
                            if detected_gesture:
                                await websocket.send_json({
                                    "type": "detection",
                                    "gesture": detected_gesture,
                                    "confidence": max_conf
                                })
                except json.JSONDecodeError:
                    pass
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)


# Needed for asyncio.wait_for
import asyncio

# ---------------------------------------------------------------------------
# Developer API Integration Endpoints
# ---------------------------------------------------------------------------
import uuid
import json

KEYS_FILE = Path("logs/api_keys.json")
KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)

def load_api_keys():
    if not KEYS_FILE.exists():
        default_keys = [
            {
                "id": "key_demo_default",
                "key": "sk_live_sign2speech_demo12345",
                "name": "Sign2Speech Sandbox App",
                "created_at": datetime.now().isoformat(),
                "calls_count": 0
            }
        ]
        with open(KEYS_FILE, "w") as f:
            json.dump(default_keys, f, indent=2)
        return default_keys
    try:
        with open(KEYS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_api_keys(keys):
    with open(KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=2)

def increment_key_usage(api_key: str):
    keys = load_api_keys()
    for k in keys:
        if k["key"] == api_key:
            k["calls_count"] = k.get("calls_count", 0) + 1
            save_api_keys(keys)
            break

async def get_developer_api_key(x_api_key: Optional[str] = Header(None)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header is missing")
    keys = load_api_keys()
    active_keys = [k["key"] for k in keys]
    if x_api_key not in active_keys:
        raise HTTPException(status_code=403, detail="Invalid X-API-Key")
    increment_key_usage(x_api_key)
    return x_api_key

class CreateKeyRequest(BaseModel):
    name: str

class DeveloperTranslateRequest(BaseModel):
    gestures: list[str]
    use_context_nlp: bool = True

class DeveloperSynthesizeRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "en-US-AriaNeural"

@app.get("/api/v1/developer/keys")
def list_keys():
    return load_api_keys()

@app.post("/api/v1/developer/keys")
def create_key(req: CreateKeyRequest):
    keys = load_api_keys()
    new_key = {
        "id": f"key_{uuid.uuid4().hex[:8]}",
        "key": f"sk_live_{uuid.uuid4().hex[:16]}",
        "name": req.name,
        "created_at": datetime.now().isoformat(),
        "calls_count": 0
    }
    keys.append(new_key)
    save_api_keys(keys)
    return new_key

@app.delete("/api/v1/developer/keys/{key_id}")
def delete_key(key_id: str):
    keys = load_api_keys()
    updated_keys = [k for k in keys if k["id"] != key_id]
    if len(keys) == len(updated_keys):
        raise HTTPException(status_code=404, detail="API Key not found")
    save_api_keys(updated_keys)
    return {"message": "API Key revoked successfully"}

@app.post("/api/v1/translate")
async def developer_translate(req: DeveloperTranslateRequest, key: str = Depends(get_developer_api_key)):
    sentence = " ".join(req.gestures)
    refined_sentence = sentence.title()
    
    if req.use_context_nlp and req.gestures:
        try:
            import requests
            prompt = (
                f"Combine the following sign language gestures in order into a single, "
                f"grammatically correct and natural English sentence: {', '.join(req.gestures)}. "
                f"Only return the final sentence, no other text."
            )
            payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
            response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            if response.status_code == 200:
                refined_sentence = response.json().get("response", "").strip()
                if (refined_sentence.startswith('"') and refined_sentence.endswith('"')) or (refined_sentence.startswith("'") and refined_sentence.endswith("'")):
                    refined_sentence = refined_sentence[1:-1]
        except Exception as e:
            logger.warning("Ollama translation failed, falling back: %s", e)
            
    return {
        "raw_gestures": req.gestures,
        "refined_sentence": refined_sentence,
        "model_used": "Local Context Refinement" if req.use_context_nlp else "Simple Join Pipeline",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/synthesize")
async def developer_synthesize(req: DeveloperSynthesizeRequest, key: str = Depends(get_developer_api_key)):
    from components.tts_engine import TTSEngine
    tts = TTSEngine(output_dir=os.getenv("TTS_OUTPUT_DIR", "logs/audio_outputs"), voice_id=req.voice_id)
    audio_path = tts.synthesize_speech(req.text, play_audio=False)
    
    if audio_path and os.path.exists(audio_path):
        return {
            "text": req.text,
            "audio_url": f"/api/audio/{os.path.basename(audio_path)}",
            "backend": tts.active_backend,
            "voice_id": req.voice_id,
            "timestamp": datetime.now().isoformat()
        }
    raise HTTPException(status_code=500, detail="Voice synthesis failed")


