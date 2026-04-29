FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and pyttsx3
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    espeak \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies from the parent requirements (mapped to /tmp here or just copied)
COPY requirements.txt /app/requirements.txt
COPY backend/requirements.txt /app/backend/requirements.txt

RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy main pipeline code and backend
COPY . /app

EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
