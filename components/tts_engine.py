#!/usr/bin/env python3
"""
Text-to-Speech Engine Module
Converts text sentences to speech audio using multiple TTS backends
"""

import os
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
import threading
from pathlib import Path

# TTS Backend imports
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    print("⚠️ pyttsx3 not available")
    PYTTSX3_AVAILABLE = False

try:
    import soundfile as sf
    import numpy as np
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    print("⚠️ Audio processing libraries not available")
    AUDIO_PROCESSING_AVAILABLE = False


class TTSEngine:
    """Text-to-Speech engine with multiple backend support"""
    
    def __init__(self, 
                 backend='pyttsx3',
                 voice_id=None,
                 speech_rate=150,
                 volume=0.9,
                 output_dir='logs/audio_outputs'):
        """
        Initialize TTS engine
        
        Args:
            backend: TTS backend ('pyttsx3')
            voice_id: Voice ID/name to use
            speech_rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
            output_dir: Directory to save audio files
        """
        self.backend = backend
        self.voice_id = voice_id
        self.speech_rate = speech_rate
        self.volume = volume
        self.output_dir = Path(output_dir)
        self.active_backend = None
        self.engine = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._init_pyttsx3()
        
        print(f"🔊 TTS Engine initialized with backend: {self.active_backend}")
    
    def _init_pyttsx3(self):
        """Initialize pyttsx3 engine"""
        if not PYTTSX3_AVAILABLE:
            print("❌ pyttsx3 not available")
            self.engine = None
            self.active_backend = None
            return
        
        try:
            self.engine = pyttsx3.init()
            
            # Set properties
            self.engine.setProperty('rate', self.speech_rate)
            self.engine.setProperty('volume', self.volume)
            
            # Set voice if specified
            if self.voice_id is not None:
                self.engine.setProperty('voice', self.voice_id)
            
            self.active_backend = 'pyttsx3'
            print("✅ pyttsx3 engine initialized")
            
        except Exception as e:
            print(f"❌ pyttsx3 initialization failed: {e}")
            self.engine = None
    
    def synthesize_speech(self, text: str, play_audio: bool = True) -> Optional[str]:
        """
        Convert text to speech and save as audio file
        
        Args:
            text: Text to convert to speech
            play_audio: Whether to play the audio immediately
            
        Returns:
            Path to saved audio file, or None if failed
        """
        if not text or not text.strip():
            return None
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"tts_{timestamp}.wav"
        audio_path = self.output_dir / audio_filename
        
        try:
            if self.active_backend == 'pyttsx3' and self.engine:
                return self._synthesize_with_pyttsx3(text, str(audio_path), play_audio)
            else:
                print("❌ No TTS backend available")
                return None
                
        except Exception as e:
            print(f"❌ TTS synthesis error: {e}")
            return None
    
    def _synthesize_with_pyttsx3(self, text: str, audio_path: str, play_audio: bool) -> Optional[str]:
        """Synthesize speech using pyttsx3"""
        try:
            print(f"🎤 Synthesizing with pyttsx3: '{text}'")
            
            # pyttsx3 doesn't directly save to file, so we'll use a workaround
            # Save to file using the engine's save_to_file method
            self.engine.save_to_file(text, audio_path)
            self.engine.runAndWait()
            
            if play_audio:
                # Also speak it directly for immediate playback
                threading.Thread(target=self._speak_with_pyttsx3, args=(text,), daemon=True).start()
            
            return audio_path
            
        except Exception as e:
            print(f"❌ pyttsx3 synthesis error: {e}")
            # Fallback: just speak without saving
            if play_audio:
                self._speak_with_pyttsx3(text)
            return None
    
    def _speak_with_pyttsx3(self, text: str):
        """Speak text directly using pyttsx3"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"❌ pyttsx3 speech error: {e}")
    
    def _play_audio_file(self, audio_path: str):
        """Play audio file (platform-specific)"""
        try:
            import platform
            system = platform.system()
            
            if system == "Windows":
                os.system(f'start "" "{audio_path}"')
            elif system == "Darwin":  # macOS
                os.system(f'afplay "{audio_path}"')
            elif system == "Linux":
                os.system(f'aplay "{audio_path}"')
            else:
                print(f"⚠️ Audio playback not supported on {system}")
                
        except Exception as e:
            print(f"❌ Audio playback error: {e}")
    
    def speak_immediately(self, text: str):
        """Speak text immediately without saving to file"""
        if not text or not text.strip():
            return
        
        try:
            if self.active_backend == 'pyttsx3' and self.engine:
                threading.Thread(target=self._speak_with_pyttsx3, args=(text,), daemon=True).start()
            else:
                print(f"🔊 Would speak: '{text}'")
                
        except Exception as e:
            print(f"❌ Immediate speech error: {e}")
    
    def set_voice_properties(self, rate: Optional[int] = None, 
                           volume: Optional[float] = None,
                           voice_id: Optional[str] = None):
        """Update voice properties"""
        if rate is not None:
            self.speech_rate = rate
        if volume is not None:
            self.volume = max(0.0, min(1.0, volume))
        if voice_id is not None:
            self.voice_id = voice_id
        
        # Update pyttsx3 engine if available
        if self.engine:
            try:
                if rate is not None:
                    self.engine.setProperty('rate', self.speech_rate)
                if volume is not None:
                    self.engine.setProperty('volume', self.volume)
                if voice_id is not None:
                    voices = self.engine.getProperty('voices')
                    for voice in voices:
                        if voice_id.lower() in voice.name.lower():
                            self.engine.setProperty('voice', voice.id)
                            break
            except Exception as e:
                print(f"❌ Error updating voice properties: {e}")
        
        print(f"🔧 Voice properties updated: rate={self.speech_rate}, volume={self.volume}")
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices"""
        voices = []
        
        if self.engine:
            try:
                pyttsx3_voices = self.engine.getProperty('voices')
                for voice in pyttsx3_voices:
                    voices.append({
                        'id': voice.id,
                        'name': voice.name,
                        'language': getattr(voice, 'languages', ['unknown'])[0],
                        'backend': 'pyttsx3'
                    })
            except Exception as e:
                print(f"❌ Error getting pyttsx3 voices: {e}")
        
        return voices
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the TTS engine"""
        return {
            'active_backend': self.active_backend,
            'available_backends': {
                'pyttsx3': self.engine is not None
            },
            'settings': {
                'speech_rate': self.speech_rate,
                'volume': self.volume,
                'voice_id': self.voice_id,
                'output_dir': str(self.output_dir)
            },
            'audio_files_count': len(list(self.output_dir.glob('*.wav')))
        }
    
    def cleanup_old_files(self, max_files: int = 100):
        """Clean up old audio files to save disk space"""
        try:
            audio_files = sorted(self.output_dir.glob('*.wav'), key=os.path.getmtime)
            
            if len(audio_files) > max_files:
                files_to_delete = audio_files[:-max_files]
                for file_path in files_to_delete:
                    file_path.unlink()
                
                print(f"🧹 Cleaned up {len(files_to_delete)} old audio files")
                
        except Exception as e:
            print(f"❌ Cleanup error: {e}")


def test_tts_engine():
    """Test function for TTS engine"""
    print("🧪 Testing TTS Engine...")
    
    # Create TTS engine
    tts = TTSEngine(backend='auto')
    
    # Test sentences
    test_sentences = [
        "Hello, this is a test.",
        "I love you.",
        "Thank you for your help.",
        "How are you today?",
        "Nice to meet you."
    ]
    
    print("\n🎤 Testing speech synthesis:")
    for i, sentence in enumerate(test_sentences, 1):
        print(f"  {i}. Testing: '{sentence}'")
        audio_path = tts.synthesize_speech(sentence, play_audio=False)
        if audio_path:
            print(f"     ✅ Saved to: {audio_path}")
        else:
            print(f"     ❌ Failed to synthesize")
        
        time.sleep(0.5)  # Brief pause between tests
    
    # Test immediate speech
    print("\n🔊 Testing immediate speech:")
    tts.speak_immediately("This is immediate speech test.")
    
    # Get engine info
    info = tts.get_engine_info()
    print(f"\n📊 Engine Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Get available voices
    voices = tts.get_available_voices()
    print(f"\n🎭 Available Voices ({len(voices)}):")
    for voice in voices[:5]:  # Show first 5 voices
        print(f"  - {voice['name']} ({voice['backend']})")


if __name__ == "__main__":
    test_tts_engine()