#!/usr/bin/env python3
"""
System Logger Module
Handles logging of recognized sentences, audio outputs, and system events
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
from collections import deque

try:
    import coloredlogs
    import logging
    COLORED_LOGS_AVAILABLE = True
except ImportError:
    import logging
    COLORED_LOGS_AVAILABLE = False


class SystemLogger:
    """Comprehensive logging system for the sign language to speech application"""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the logging system
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.audio_dir = self.log_dir / "audio_outputs"
        self.audio_dir.mkdir(exist_ok=True)
        
        # Log files
        self.text_log_file = self.log_dir / "recognized_text.txt"
        self.system_log_file = self.log_dir / "system.log"
        self.session_log_file = self.log_dir / "session_data.json"
        
        # Session data
        self.session_start = datetime.now()
        self.session_data = {
            'session_id': self.session_start.strftime("%Y%m%d_%H%M%S"),
            'start_time': self.session_start.isoformat(),
            'recognized_sentences': [],
            'audio_files': [],
            'system_events': [],
            'statistics': {
                'total_sentences': 0,
                'total_audio_files': 0,
                'session_duration': 0,
                'average_sentence_length': 0
            }
        }
        
        # Thread-safe logging
        self.log_lock = threading.Lock()
        self.log_queue = deque()
        
        # Setup logging
        self._setup_logging()
        
        # Start background logging thread
        self.logging_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.logging_thread.start()
        
        self.log_system_event("Logger initialized", "INFO")
        print(f"📝 System Logger initialized - Session ID: {self.session_data['session_id']}")
    
    def _setup_logging(self):
        """Setup Python logging configuration"""
        # Create logger
        self.logger = logging.getLogger('SignLanguageSystem')
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(self.system_log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Setup colored logs if available
        if COLORED_LOGS_AVAILABLE:
            coloredlogs.install(
                level='INFO',
                logger=self.logger,
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def _log_worker(self):
        """Background worker for processing log queue"""
        while True:
            try:
                if self.log_queue:
                    log_entry = self.log_queue.popleft()
                    self._process_log_entry(log_entry)
                else:
                    threading.Event().wait(0.1)  # Small delay when queue is empty
            except Exception as e:
                print(f"❌ Log worker error: {e}")
    
    def _process_log_entry(self, log_entry: Dict[str, Any]):
        """Process a single log entry"""
        try:
            entry_type = log_entry.get('type')
            
            if entry_type == 'sentence':
                self._write_sentence_log(log_entry)
            elif entry_type == 'audio':
                self._log_audio_file(log_entry)
            elif entry_type == 'system':
                self._write_system_log(log_entry)
            
            # Update session data
            self._update_session_data(log_entry)
            
        except Exception as e:
            print(f"❌ Error processing log entry: {e}")
    
    def log_sentence(self, sentence: str, confidence: float = 1.0, 
                    track_id: Optional[int] = None, 
                    sign_sequence: Optional[List[str]] = None):
        """Log a recognized sentence"""
        timestamp = datetime.now()
        
        log_entry = {
            'type': 'sentence',
            'timestamp': timestamp.isoformat(),
            'sentence': sentence,
            'confidence': confidence,
            'track_id': track_id,
            'sign_sequence': sign_sequence or [],
            'session_id': self.session_data['session_id']
        }
        
        with self.log_lock:
            self.log_queue.append(log_entry)
        
        print(f"📝 Logged sentence: '{sentence}'")
    
    def log_audio_file(self, audio_path: str, sentence: str, 
                      file_size: Optional[int] = None):
        """Log an audio file creation"""
        timestamp = datetime.now()
        
        # Get file size if not provided
        if file_size is None and os.path.exists(audio_path):
            file_size = os.path.getsize(audio_path)
        
        log_entry = {
            'type': 'audio',
            'timestamp': timestamp.isoformat(),
            'audio_path': audio_path,
            'sentence': sentence,
            'file_size': file_size,
            'session_id': self.session_data['session_id']
        }
        
        with self.log_lock:
            self.log_queue.append(log_entry)
        
        print(f"🔊 Logged audio file: {os.path.basename(audio_path)}")
    
    def log_system_event(self, message: str, level: str = "INFO", 
                        event_type: str = "general", 
                        additional_data: Optional[Dict] = None):
        """Log a system event"""
        timestamp = datetime.now()
        
        log_entry = {
            'type': 'system',
            'timestamp': timestamp.isoformat(),
            'level': level.upper(),
            'message': message,
            'event_type': event_type,
            'additional_data': additional_data or {},
            'session_id': self.session_data['session_id']
        }
        
        with self.log_lock:
            self.log_queue.append(log_entry)
        
        # Also log to Python logger
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.log(log_level, f"[{event_type}] {message}")
    
    def _write_sentence_log(self, log_entry: Dict[str, Any]):
        """Write sentence to text log file"""
        try:
            timestamp_str = datetime.fromisoformat(log_entry['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
            
            log_line = f"[{timestamp_str}] {log_entry['sentence']}"
            
            if log_entry.get('track_id') is not None:
                log_line += f" (Track: {log_entry['track_id']})"
            
            if log_entry.get('confidence', 1.0) < 1.0:
                log_line += f" (Confidence: {log_entry['confidence']:.2f})"
            
            if log_entry.get('sign_sequence'):
                log_line += f" [Signs: {', '.join(log_entry['sign_sequence'])}]"
            
            log_line += "\n"
            
            with open(self.text_log_file, 'a', encoding='utf-8') as f:
                f.write(log_line)
                
        except Exception as e:
            print(f"❌ Error writing sentence log: {e}")
    
    def _log_audio_file(self, log_entry: Dict[str, Any]):
        """Process audio file log entry"""
        # Audio files are already saved by TTS engine
        # This just updates our tracking
        pass
    
    def _write_system_log(self, log_entry: Dict[str, Any]):
        """Write system event to log"""
        # System events are handled by Python logger
        pass
    
    def _update_session_data(self, log_entry: Dict[str, Any]):
        """Update session data with new log entry"""
        try:
            entry_type = log_entry.get('type')
            
            if entry_type == 'sentence':
                self.session_data['recognized_sentences'].append({
                    'timestamp': log_entry['timestamp'],
                    'sentence': log_entry['sentence'],
                    'confidence': log_entry.get('confidence', 1.0),
                    'track_id': log_entry.get('track_id'),
                    'sign_sequence': log_entry.get('sign_sequence', [])
                })
                self.session_data['statistics']['total_sentences'] += 1
                
            elif entry_type == 'audio':
                self.session_data['audio_files'].append({
                    'timestamp': log_entry['timestamp'],
                    'path': log_entry['audio_path'],
                    'sentence': log_entry['sentence'],
                    'file_size': log_entry.get('file_size')
                })
                self.session_data['statistics']['total_audio_files'] += 1
                
            elif entry_type == 'system':
                self.session_data['system_events'].append({
                    'timestamp': log_entry['timestamp'],
                    'level': log_entry['level'],
                    'message': log_entry['message'],
                    'event_type': log_entry['event_type']
                })
            
            # Update statistics
            self._update_statistics()
            
            # Save session data periodically
            if len(self.session_data['recognized_sentences']) % 5 == 0:
                self._save_session_data()
                
        except Exception as e:
            print(f"❌ Error updating session data: {e}")
    
    def _update_statistics(self):
        """Update session statistics"""
        try:
            # Session duration
            current_time = datetime.now()
            duration = (current_time - self.session_start).total_seconds()
            self.session_data['statistics']['session_duration'] = duration
            
            # Average sentence length
            sentences = self.session_data['recognized_sentences']
            if sentences:
                total_length = sum(len(s['sentence'].split()) for s in sentences)
                avg_length = total_length / len(sentences)
                self.session_data['statistics']['average_sentence_length'] = round(avg_length, 2)
                
        except Exception as e:
            print(f"❌ Error updating statistics: {e}")
    
    def _save_session_data(self):
        """Save session data to JSON file"""
        try:
            with open(self.session_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving session data: {e}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        self._update_statistics()
        
        return {
            'session_id': self.session_data['session_id'],
            'start_time': self.session_data['start_time'],
            'duration_minutes': round(self.session_data['statistics']['session_duration'] / 60, 2),
            'total_sentences': self.session_data['statistics']['total_sentences'],
            'total_audio_files': self.session_data['statistics']['total_audio_files'],
            'average_sentence_length': self.session_data['statistics']['average_sentence_length'],
            'recent_sentences': self.session_data['recognized_sentences'][-5:] if self.session_data['recognized_sentences'] else []
        }
    
    def get_log_files_info(self) -> Dict[str, Any]:
        """Get information about log files"""
        info = {
            'log_directory': str(self.log_dir),
            'files': {}
        }
        
        # Check each log file
        log_files = {
            'recognized_text': self.text_log_file,
            'system_log': self.system_log_file,
            'session_data': self.session_log_file
        }
        
        for name, file_path in log_files.items():
            if file_path.exists():
                stat = file_path.stat()
                info['files'][name] = {
                    'path': str(file_path),
                    'size_bytes': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
            else:
                info['files'][name] = {'path': str(file_path), 'exists': False}
        
        # Audio files info
        audio_files = list(self.audio_dir.glob('*.wav'))
        info['audio_files'] = {
            'count': len(audio_files),
            'total_size_bytes': sum(f.stat().st_size for f in audio_files),
            'directory': str(self.audio_dir)
        }
        
        return info
    
    def cleanup_old_logs(self, days_to_keep: int = 7):
        """Clean up old log files"""
        try:
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
            
            # Clean up old audio files
            audio_files = list(self.audio_dir.glob('*.wav'))
            deleted_count = 0
            
            for audio_file in audio_files:
                if audio_file.stat().st_mtime < cutoff_time:
                    audio_file.unlink()
                    deleted_count += 1
            
            if deleted_count > 0:
                self.log_system_event(f"Cleaned up {deleted_count} old audio files", "INFO", "cleanup")
                
        except Exception as e:
            self.log_system_event(f"Error during cleanup: {e}", "ERROR", "cleanup")
    
    def finalize_session(self):
        """Finalize the current session"""
        try:
            # Save final session data
            self.session_data['end_time'] = datetime.now().isoformat()
            self._update_statistics()
            self._save_session_data()
            
            # Log session summary
            summary = self.get_session_summary()
            self.log_system_event(
                f"Session completed - {summary['total_sentences']} sentences, {summary['duration_minutes']} minutes",
                "INFO",
                "session_end",
                summary
            )
            
            print(f"📊 Session finalized: {summary['total_sentences']} sentences in {summary['duration_minutes']} minutes")
            
        except Exception as e:
            print(f"❌ Error finalizing session: {e}")


def test_logger():
    """Test function for system logger"""
    print("🧪 Testing System Logger...")
    
    # Create logger
    logger = SystemLogger("test_logs")
    
    # Test sentence logging
    test_sentences = [
        "Hello, this is a test sentence.",
        "I love you.",
        "Thank you for your help.",
        "How are you today?"
    ]
    
    print("\n📝 Testing sentence logging:")
    for i, sentence in enumerate(test_sentences):
        logger.log_sentence(
            sentence=sentence,
            confidence=0.8 + (i * 0.05),
            track_id=i % 2,
            sign_sequence=["test", "sign", "sequence"]
        )
    
    # Test audio file logging
    print("\n🔊 Testing audio file logging:")
    for i, sentence in enumerate(test_sentences[:2]):
        audio_path = f"test_logs/audio_outputs/test_audio_{i}.wav"
        logger.log_audio_file(audio_path, sentence, 1024 * (i + 1))
    
    # Test system events
    print("\n🔧 Testing system event logging:")
    logger.log_system_event("System started", "INFO", "startup")
    logger.log_system_event("Model loaded successfully", "INFO", "model")
    logger.log_system_event("Warning: Low confidence detection", "WARNING", "detection")
    
    # Get session summary
    import time
    time.sleep(1)  # Allow time for background processing
    
    summary = logger.get_session_summary()
    print(f"\n📊 Session Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Get log files info
    files_info = logger.get_log_files_info()
    print(f"\n📁 Log Files Info:")
    for key, value in files_info.items():
        print(f"  {key}: {value}")
    
    # Finalize session
    logger.finalize_session()


if __name__ == "__main__":
    test_logger()