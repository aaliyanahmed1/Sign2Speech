import sys
import os
from fastapi import FastAPI, File, UploadFile, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import cv2

# Add parent directory to path so we can import components/utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from components.deep_sort_tracker import DeepSORTTracker
from components.tts_engine import TTSEngine

# Define valid class mappings directly (1-indexed mapping)
CLASS_NAMES = [
    "school", "sorry", "help", "easy", "work",
    "age", "effort", "respect", "near", "home",
    "friend", "washroom", "preset", "pass", "fail",
    "village", "eating", "drinking", "teacher", "dress",
    "message", "good"
]

app = FastAPI(title="Sign2Speech API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SpeakRequest(BaseModel):
    sentence: str

@app.get("/")
def read_root():
    return {"status": "Sign2Speech API is running"}

import numpy as np
from components.yolo2voice_pipeline import yolo_classes_to_voice

# Initialize raw Ultralytics model directly as requested
try:
    model = YOLO("models/sign.pt")
except Exception as e:
    print(f"Failed to load raw YOLO model: {e}")
    model = None

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"error": "Could not decode image"}

    gestures = []
    sentence = "No gesture detected"
    
    if model:
        # Run pure ultralytics inference
        results = model.predict(frame, conf=0.1) # low conf to ensure we catch it
        
        for r in results:
            if r.boxes:
                for cls_id in r.boxes.cls.tolist():
                    raw_name = model.names[int(cls_id)]
                    # Attempt to parse integer suffix from "class_22", "22", or fall back 
                    try:
                        # Extract numbers from the raw class string
                        import re
                        numeric_part = re.search(r'\d+', raw_name)
                        if numeric_part:
                            idx = int(numeric_part.group()) - 1
                            if 0 <= idx < len(CLASS_NAMES):
                                class_name = CLASS_NAMES[idx]
                            else:
                                class_name = raw_name
                        else:
                            # if it's already a clean string and not "class_X"
                            class_name = raw_name
                    except:
                        class_name = raw_name
                        
                    gestures.append(class_name)
                    
        print(f"Mapped matching Gestures: {gestures}")
        
        # Deduplicate to keep logical sequence base
        gestures = list(set(gestures))
        
        if gestures:
            # Removed sentence builder completely, just join the raw matched gestures cleanly
            sentence = " ".join(gestures).title()
    else:
        # Mock behavior if model doesn't load
        import random
        gestures = [random.choice(["HELLO", "YES", "NO", "THANK_YOU"])]
        sentence = f"Generated sentence for {gestures[0]}"

    return {
        "job_id": "image_sync_job",
        "gestures": gestures,
        "sentence": sentence
    }

@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    # Retrieve job result from db or memory
    return {
        "job_id": job_id,
        "gestures": [],
        "sentence": "Hello world",
        "annotated_url": "/static/annotated_" + job_id + ".mp4"
    }

@app.websocket("/api/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        # Simulate detections periodically without waiting for incoming data
        import random
        classes = ["HELLO", "THANK_YOU", "I_LOVE_YOU", "YES", "NO", "PLEASE"]
        while True:
            await asyncio.sleep(2)  # Emit every 2 seconds
            await websocket.send_json({
                "type": "detection",
                "gesture": random.choice(classes),
                "confidence": round(random.uniform(0.7, 0.99), 2)
            })
    except Exception as e:
        print("WebSocket disconnected", e)

@app.post("/api/speak")
async def speak(req: SpeakRequest):
    # Synthesize speech and return audio file path or blob
    # engine = TTSEngine()
    # path = engine.synthesize_speech(req.sentence)
    return {"audio_url": f"/static/audio_{hash(req.sentence)}.wav"}

@app.get("/api/analytics")
async def get_analytics():
    return {
        "total_sessions": 120,
        "gestures_detected": 1500,
        "avg_confidence": 0.89,
        "sentences_spoken": 340
    }

@app.get("/api/classes")
async def get_classes():
    # Example lists, replace with actual 22 classes
    return {"classes": ["HELLO", "THANK_YOU", "I_LOVE_YOU", "YES", "NO", "PLEASE"]}
