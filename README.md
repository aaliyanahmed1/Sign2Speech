# Sign2Speech Web UI

This project bridges silence and speech in real-time, converting sign language from real-time webcam feeds or pre-recorded videos into spoken output. It consists of a React/Vite frontend and a FastAPI (Python) backend wrapping an advanced AI pipeline (YOLO12 + DeepSORT + Ollama + pyttsx3).

## Stack
* **Frontend**: React, Vite, Tailwind CSS, React-Router, Framer Motion, Zustand
* **Backend**: FastAPI, Websockets, OpenCV, DeepSORT Tracker, pyttsx3

## Requirements
* Docker Desktop (or standard Docker + Compose)
* Python 3.10+ (if running locally)
* Node.js 18+ (if running locally)

## Running the Application (Docker)
The easiest way to run both the frontend and backend is using `docker-compose`:

```bash
docker-compose up --build
```
This will start:
- Frontend on http://localhost:5173
- Backend API on http://localhost:8000

## Running Locally Without Docker
### Sub-project setup
If you want to run the modules locally to debug them, follow these steps:

#### 1. Backend (FastAPI)
```bash
# In the root project directory:
python -m venv venv
# Activate the venv
source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
pip install -r requirements.txt
pip install -r backend/requirements.txt

# Start the server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
API Documentation will be available at http://localhost:8000/docs

#### 2. Frontend (Vite/React)
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
