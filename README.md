# Sign2Speech - Professional Sign Language Interpreter

This project bridges silence and speech in real-time, converting sign language from real-time webcam feeds or pre-recorded videos into spoken output. It consists of a React/Vite frontend and a FastAPI (Python) backend wrapping an advanced AI pipeline (YOLO12 + DeepSORT + Ollama + pyttsx3).

## 🚀 Features
- **Real-time Sign Language Detection**: Custom YOLO12 model trained for 22 sign language gestures
- **Multi-gesture Tracking**: DeepSORT algorithm for consistent gesture identification
- **Natural Language Processing**: Converts gesture sequences into grammatically correct sentences
- **Voice Synthesis**: Text-to-speech conversion with audio file generation
- **Web UI**: Modern React interface with Live Detection and Analytics
- **Image & Video Support**: Process both static images and video files

## 📋 Supported Sign Language Gestures
The system recognizes 22 different sign language gestures:
- **Basic**: school, sorry, help, easy, work, age, effort, respect
- **Location**: near, home, village, washroom
- **Social**: friend, teacher, message, good
- **Actions**: eating, drinking, pass, fail
- **Settings**: preset, dress

## Stack
* **Frontend**: React, Vite, Tailwind CSS, React-Router, Framer Motion, Zustand
* **Backend**: FastAPI, Websockets, OpenCV, DeepSORT Tracker, pyttsx3, Ollama

## Requirements
* Docker Desktop (or standard Docker + Compose)
* Python 3.10+ (if running locally)
* Node.js 18+ (if running locally)
* Ollama with llama3 model (if using local NLP inference)

## Running the Application (Docker)
The easiest way to run both the frontend and backend is using `docker-compose`:

```bash
docker-compose up --build
```
This will start:
- Frontend on http://localhost:5173
- Backend API on http://localhost:8000

## Running Locally Without Docker

### 1. Backend (FastAPI)
```bash
# In the root project directory:
python -m venv venv
# Activate the venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
pip install -r backend/requirements.txt

# Start the server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
API Documentation will be available at http://localhost:8000/docs

### 2. Frontend (Vite/React)
```bash
cd frontend
npm install
npm run dev
```

## Available Features
- **Live Detection**: Connects via WebSockets to stream real-time bounding boxes directly from your webcam.
- **Upload & Analyze**: Drag and drop prerecorded sign language videos to be annotated.
- **Analytics Dashboard**: Track model performance and gesture frequencies.

*(Please refer to the technical prompt or project requirements for specific architecture mappings).*
