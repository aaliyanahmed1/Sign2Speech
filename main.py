#!/usr/bin/env python3
"""
Real-time Sign Language to Speech System
Main pipeline entry point that coordinates detection, tracking, NLP, and TTS
"""

import os
import cv2
import time
import threading
from collections import defaultdict, deque
from datetime import datetime
#from sentence_transformers import SentenceTransformer
import pyttsx3

from components.yolo_inference import YOLO12Detector
from components.deep_sort_tracker import DeepSORTTracker
from components.sentence_builder import SentenceBuilder
from components.tts_engine import TTSEngine
from components.yolo2voice_pipeline import generate_sentence_with_ollama, text_to_speech
from utils.system_logger import SystemLogger
from utils.draw import DrawingUtils
from utils.video_utils import VideoProcessor
from components.yolo2voice_pipeline import yolo_classes_to_voice


class SignLanguageToSpeechSystem:
    def __init__(self, model_path="models/sign.pt", confidence_threshold=0.5):
        """
        Initialize the real-time sign language to speech system
        """
        # Core components
        self.yolo_detector = YOLO12Detector(model_path, confidence_threshold)
        self.tracker = DeepSORTTracker()
        self.sentence_builder = SentenceBuilder()
        self.tts_engine = TTSEngine()
        self.logger = SystemLogger()
        
        # Utilities
        self.drawer = DrawingUtils()
        self.video_processor = VideoProcessor()
        
        # Tracking buffers
        self.track_buffers = defaultdict(lambda: deque(maxlen=10))  # Store last 10 signs per track
        self.track_timestamps = defaultdict(float)
        self.processed_sentences = set()  # Avoid duplicate processing
        
        # Configuration
        self.buffer_timeout = 3.0  # Process buffer after 3 seconds of inactivity
        self.min_signs_for_sentence = 2  # Minimum signs needed to form sentence
        
        # Threading for TTS (non-blocking)
        self.tts_queue = deque()
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()
        
        print("🚀 Sign Language to Speech System initialized!")
    
    def _tts_worker(self):
        """Background worker for TTS processing"""
        while True:
            if self.tts_queue:
                sentence = self.tts_queue.popleft()
                try:
                    # Generate and save audio
                    audio_path = self.tts_engine.synthesize_speech(sentence)
                    print(f"🔊 Audio saved: {audio_path}")
                    
                    # Log the sentence
                    self.logger.log_sentence(sentence)
                    
                except Exception as e:
                    print(f"❌ TTS Error: {e}")
            
            time.sleep(0.1)
    
    def process_frame(self, frame):
        """
        Process a single frame through the entire pipeline
        """
        current_time = time.time()
        
        # Step 1: YOLO Detection
        detections = self.yolo_detector.detect(frame)
        
        # Step 2: DeepSORT Tracking
        tracks = self.tracker.update(detections, frame)
        
        # Step 3: Update tracking buffers
        active_track_ids = set()
        for track in tracks:
            track_id = track.track_id
            class_name = track.class_name
            confidence = track.confidence
            
            active_track_ids.add(track_id)
            
            # Add to buffer if confidence is high enough
            if confidence > 0.7:
                self.track_buffers[track_id].append({
                    'class': class_name,
                    'confidence': confidence,
                    'timestamp': current_time
                })
                self.track_timestamps[track_id] = current_time
        
        # Step 4: Process buffers for sentence generation
        self._process_track_buffers(current_time)
        
        # Step 5: Draw annotations
        annotated_frame = self.drawer.draw_tracks(frame, tracks)
        
        # Step 6: Display current sentence being built
        for track_id, buffer in self.track_buffers.items():
            if buffer:
                current_sequence = [item['class'] for item in buffer]
                partial_sentence = self.sentence_builder.build_partial_sentence(current_sequence)
                annotated_frame = self.drawer.draw_text(
                    annotated_frame, 
                    f"Track {track_id}: {partial_sentence}",
                    position=(10, 30 + track_id * 25)
                )
        
        return annotated_frame
    
    def _process_track_buffers(self, current_time):
        """
        Process tracking buffers to generate sentences using Ollama
        """
        tracks_to_process = []
        
        for track_id, buffer in self.track_buffers.items():
            if not buffer:
                continue
            
            last_update = self.track_timestamps.get(track_id, 0)
            time_since_update = current_time - last_update
            
            # Process if buffer has enough signs and hasn't been updated recently
            if (len(buffer) >= self.min_signs_for_sentence and 
                time_since_update > self.buffer_timeout):
                tracks_to_process.append(track_id)
        
        # Process eligible tracks with Ollama
        for track_id in tracks_to_process:
            buffer = self.track_buffers[track_id]
            sign_sequence = [item['class'] for item in buffer]
            unique_signs = list(set(sign_sequence))  # Remove duplicates
            
            for sign_class in unique_signs:
                try:
                    # Generate sentence using Ollama
                    sentence = generate_sentence_with_ollama(sign_class)
                    
                    if sentence and sentence.strip():
                        print(f"🎯 Generated sentence for '{sign_class}': '{sentence}'")
                        
                        # Create audio file with class name
                        audio_filename = f"logs/audio_outputs/{sign_class}_{int(current_time)}.wav"
                        os.makedirs("logs/audio_outputs", exist_ok=True)
                        
                        # Generate and save audio
                        text_to_speech(sentence, audio_filename)
                        print(f"🔊 Audio saved: {audio_filename}")
                        
                        # Also add to TTS queue for immediate playback
                        self.tts_queue.append(sentence)
                        
                except Exception as e:
                    print(f"❌ Error processing class '{sign_class}': {e}")
            
            # Clear the buffer after processing
            self.track_buffers[track_id].clear()
    
    def _generate_sentence_from_buffer(self, track_id):
        """
        Generate sentence from a track's buffer
        """
        buffer = self.track_buffers[track_id]
        if not buffer:
            return
        
        # Extract sign sequence
        sign_sequence = [item['class'] for item in buffer]
        
        # Create unique identifier for this sequence
        sequence_id = f"{track_id}_{hash(tuple(sign_sequence))}"
        
        # Skip if already processed
        if sequence_id in self.processed_sentences:
            return
        
        # Build sentence
        sentence = self.sentence_builder.build_sentence(sign_sequence)
        
        if sentence and len(sentence.strip()) > 0:
            print(f"📝 Generated sentence from track {track_id}: '{sentence}'")
            
            # Add to TTS queue
            self.tts_queue.append(sentence)
            
            # Mark as processed
            self.processed_sentences.add(sequence_id)
            
            # Clear the buffer
            self.track_buffers[track_id].clear()
    
    def run(self, source=0):
        """
        Run the real-time system
        """
        print(f"🎥 Starting video capture from source: {source}")
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("❌ Error: Could not open video source")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🔴 System running! Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    processed_frame = self.drawer.draw_text(
                        processed_frame,
                        f"FPS: {fps:.1f}",
                        position=(10, processed_frame.shape[0] - 30)
                    )
                
                # Display frame
                cv2.imshow('Sign Language to Speech System', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"logs/frame_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"💾 Frame saved: {filename}")
        
        except KeyboardInterrupt:
            print("\n⏹️ System stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("🏁 System shutdown complete")


def process_and_generate_audio(input_path):
    # Load image (for video, adapt as needed)
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"Could not load image: {input_path}")
        return "Error: Could not load image."

    # 1. Detect classes
    # Use absolute path for the YOLO model
    detector = YOLO12Detector("F:/NLPPRO/sign2speech/models/sign.pt")
    detections = detector.detect(frame)
    detected_classes = list(set([det.class_name for det in detections]))
    print("Detected classes:", detected_classes)

    if not detected_classes:
        return "No classes detected."

    # 2. Generate sentences and audio using Ollama + pyttsx3
    yolo_classes_to_voice(detected_classes)
    return f"Processed {len(detected_classes)} classes. Audio saved in 'voices/'."


# Example usage:
if __name__ == "__main__":
    result = process_and_generate_audio("F:/NLPPRO/sign2speech/dataset/IMG_6633.jpg")
    print(result)
    
    # Main entry point
    print("🎯 Initializing Sign Language to Speech System...")
    
    # Initialize system
    system = SignLanguageToSpeechSystem(
        model_path="models/sign.pt",
        confidence_threshold=0.5
    )
    
    # Run the system
    system.run(source=0)  # Use webcam (0) or video file path