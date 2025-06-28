# Logs Directory

This directory contains all output logs and generated files from the Sign Language to Speech System.

## Directory Structure

```
logs/
├── README.md                    # This file
├── recognized_text.txt          # All recognized sentences with timestamps
├── session_data.json           # Current session metadata and statistics
├── system.log                  # System events and debug information
└── audio_outputs/              # Generated speech audio files
    ├── tts_YYYYMMDD_HHMMSS.wav # Individual audio files
    └── README.md               # Audio files documentation
```

## File Descriptions

### `recognized_text.txt`
Contains all recognized sentences with detailed information:
```
[2024-01-15 14:30:25] I love you (Track: 1) (Confidence: 0.92) [Signs: I, love, you]
[2024-01-15 14:30:32] Thank you (Track: 2) (Confidence: 0.85) [Signs: thank_you]
[2024-01-15 14:30:45] How are you? (Track: 1) [Signs: how, are, you]
```

**Format**: `[timestamp] sentence (Track: ID) (Confidence: score) [Signs: sign1, sign2, ...]`

### `session_data.json`
JSON file containing session metadata and statistics:
```json
{
  "session_id": "20240115_143020",
  "start_time": "2024-01-15T14:30:20.123456",
  "end_time": null,
  "recognized_sentences": [
    {
      "timestamp": "2024-01-15T14:30:25.456789",
      "sentence": "I love you",
      "track_id": 1,
      "confidence": 0.92,
      "signs": ["I", "love", "you"],
      "audio_file": "tts_20240115_143025.wav"
    }
  ],
  "audio_files": [
    {
      "filename": "tts_20240115_143025.wav",
      "sentence": "I love you",
      "timestamp": "2024-01-15T14:30:25.456789",
      "duration": 2.5,
      "file_size": 123456
    }
  ],
  "statistics": {
    "total_sentences": 15,
    "total_audio_files": 15,
    "session_duration": 300.5,
    "average_sentence_length": 3.2,
    "most_common_signs": {
      "I": 8,
      "love": 5,
      "you": 7
    },
    "tracks_processed": 3
  }
}
```

### `system.log`
System events, errors, and debug information:
```
2024-01-15 14:30:20,123 - INFO - System initialized successfully
2024-01-15 14:30:20,456 - INFO - Camera opened: resolution 1280x720, FPS: 30
2024-01-15 14:30:21,789 - DEBUG - YOLO model loaded: yolov8_sign.pt
2024-01-15 14:30:22,012 - WARNING - DeepSORT model not found, using simple tracker
2024-01-15 14:30:25,345 - INFO - Sentence recognized: "I love you" (Track: 1)
2024-01-15 14:30:25,678 - INFO - Audio generated: tts_20240115_143025.wav
```

## Log Management

### Automatic Cleanup
The system automatically manages log files:
- **Audio files**: Cleaned up after 7 days (configurable)
- **Session data**: Archived after 30 days
- **System logs**: Rotated when they exceed 10MB

### Manual Cleanup
```bash
# Remove old audio files (older than 7 days)
find audio_outputs/ -name "*.wav" -mtime +7 -delete

# Archive old session data
mv session_data.json session_data_$(date +%Y%m%d).json

# Clear recognized text (backup first!)
cp recognized_text.txt recognized_text_backup_$(date +%Y%m%d).txt
> recognized_text.txt
```

### Log Analysis

#### Python Script for Log Analysis
```python
import json
from datetime import datetime
from collections import Counter

def analyze_session_logs():
    with open('session_data.json', 'r') as f:
        data = json.load(f)
    
    stats = data['statistics']
    print(f"Session Duration: {stats['session_duration']:.1f} seconds")
    print(f"Total Sentences: {stats['total_sentences']}")
    print(f"Average Sentence Length: {stats['average_sentence_length']:.1f} signs")
    print(f"Most Common Signs: {stats['most_common_signs']}")
    
    # Analyze sentence patterns
    sentences = [item['sentence'] for item in data['recognized_sentences']]
    sentence_counter = Counter(sentences)
    print(f"Most Common Sentences: {sentence_counter.most_common(5)}")

if __name__ == "__main__":
    analyze_session_logs()
```

#### Bash Script for Quick Stats
```bash
#!/bin/bash
# quick_stats.sh

echo "=== Sign Language System Log Summary ==="
echo "Total sentences: $(wc -l < recognized_text.txt)"
echo "Audio files: $(ls audio_outputs/*.wav 2>/dev/null | wc -l)"
echo "Log file size: $(du -h system.log 2>/dev/null | cut -f1)"
echo "Session data size: $(du -h session_data.json 2>/dev/null | cut -f1)"
echo "Last activity: $(tail -1 recognized_text.txt | cut -d']' -f1 | tr -d '[')"
```

## Configuration

### Logging Levels
Adjust in `components/logger.py`:
```python
# Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
logging.basicConfig(level=logging.INFO)

# For development/debugging
logging.basicConfig(level=logging.DEBUG)

# For production (less verbose)
logging.basicConfig(level=logging.WARNING)
```

### File Rotation
```python
from logging.handlers import RotatingFileHandler

# Rotate logs when they reach 10MB, keep 5 backup files
handler = RotatingFileHandler(
    'logs/system.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
```

### Custom Log Formats
```python
# Detailed format for debugging
detailed_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'

# Simple format for production
simple_format = '%(asctime)s - %(levelname)s - %(message)s'
```

## Troubleshooting

### Common Issues

1. **Permission Errors**:
   ```bash
   # Ensure write permissions
   chmod 755 logs/
   chmod 644 logs/*.txt logs/*.json
   ```

2. **Disk Space**:
   ```bash
   # Check available space
   df -h .
   
   # Clean up old files
   find logs/ -name "*.wav" -mtime +30 -delete
   ```

3. **Corrupted JSON**:
   ```python
   # Validate session data
   import json
   try:
       with open('logs/session_data.json', 'r') as f:
           data = json.load(f)
       print("JSON is valid")
   except json.JSONDecodeError as e:
       print(f"JSON error: {e}")
   ```

### Recovery Procedures

1. **Restore from Backup**:
   ```bash
   # If session_data.json is corrupted
   cp session_data_backup.json session_data.json
   ```

2. **Rebuild Session Data**:
   ```python
   # Rebuild from recognized_text.txt
   from components.logger import SystemLogger
   logger = SystemLogger()
   logger.rebuild_session_from_text_log()
   ```

## Privacy and Security

### Data Protection
- **No personal data**: Only sign language text and audio
- **Local storage**: All data stays on your machine
- **Encryption**: Consider encrypting sensitive logs

### Secure Deletion
```bash
# Secure delete (Linux/Mac)
shred -vfz -n 3 logs/recognized_text.txt

# Windows
sdelete -p 3 -s -z logs/
```

---

**Note**: This directory will be automatically created and populated when you run the system. All timestamps are in local time zone.