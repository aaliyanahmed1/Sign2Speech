# Audio Outputs Directory

This directory contains all generated speech audio files from the Sign Language to Speech System.

## File Naming Convention

Audio files are automatically named using the following pattern:
```
tts_YYYYMMDD_HHMMSS.wav
```

**Examples**:
- `tts_20240115_143025.wav` - Generated on Jan 15, 2024 at 14:30:25
- `tts_20240115_143032.wav` - Generated on Jan 15, 2024 at 14:30:32
- `tts_20240115_143045.wav` - Generated on Jan 15, 2024 at 14:30:45

## Audio File Properties

### Default Settings
- **Format**: WAV (uncompressed)
- **Sample Rate**: 22050 Hz (configurable)
- **Bit Depth**: 16-bit
- **Channels**: Mono
- **Duration**: Varies based on sentence length (typically 1-5 seconds)

### File Size Estimates
- **Short sentence** (1-2 words): ~50-100 KB
- **Medium sentence** (3-5 words): ~100-200 KB
- **Long sentence** (6+ words): ~200-500 KB

## TTS Backend Configurations

### pyttsx3 Backend
```python
# Configuration in tts_engine.py
engine.setProperty('rate', 150)      # Words per minute
engine.setProperty('volume', 0.9)    # Volume (0.0-1.0)
engine.setProperty('voice', voice_id) # Voice selection
```

**Available Voices** (Windows):
- Microsoft David Desktop
- Microsoft Zira Desktop
- Microsoft Mark Desktop

### Coqui TTS Backend
```python
# High-quality neural TTS
model_name = "tts_models/en/ljspeech/tacotron2-DDC"
sample_rate = 22050
```

**Available Models**:
- `tacotron2-DDC`: Fast, good quality
- `glow-tts`: Very fast, decent quality
- `speedy-speech`: Fastest, basic quality

## Audio Quality Settings

### High Quality (Recommended)
```python
TTSEngine(
    backend='coqui',
    sample_rate=22050,
    quality='high'
)
```

### Fast Generation (Real-time)
```python
TTSEngine(
    backend='pyttsx3',
    speech_rate=180,
    quality='fast'
)
```

### Custom Settings
```python
TTSEngine(
    backend='auto',
    speech_rate=150,        # Words per minute
    volume=0.9,            # Volume level
    pitch=0,               # Pitch adjustment
    sample_rate=22050,     # Audio sample rate
    output_format='wav'    # Output format
)
```

## File Management

### Automatic Cleanup
The system automatically manages audio files:

```python
# In logger.py - cleanup old files
def cleanup_old_audio_files(self, days_to_keep=7):
    cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
    for file_path in self.audio_output_dir.glob("tts_*.wav"):
        if file_path.stat().st_mtime < cutoff_time:
            file_path.unlink()
```

### Manual Cleanup Commands

```bash
# Remove files older than 7 days (Linux/Mac)
find . -name "tts_*.wav" -mtime +7 -delete

# Remove files older than 7 days (Windows PowerShell)
Get-ChildItem -Path . -Name "tts_*.wav" | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-7)} | Remove-Item

# Remove all audio files (backup first!)
rm tts_*.wav

# Archive old files
mkdir archive_$(date +%Y%m%d)
mv tts_*.wav archive_$(date +%Y%m%d)/
```

## Audio Analysis

### Python Script for Audio Stats
```python
import os
import wave
from pathlib import Path

def analyze_audio_files():
    audio_dir = Path('.')
    wav_files = list(audio_dir.glob('tts_*.wav'))
    
    total_files = len(wav_files)
    total_size = sum(f.stat().st_size for f in wav_files)
    total_duration = 0
    
    for wav_file in wav_files:
        try:
            with wave.open(str(wav_file), 'rb') as w:
                frames = w.getnframes()
                rate = w.getframerate()
                duration = frames / rate
                total_duration += duration
        except:
            continue
    
    print(f"Total audio files: {total_files}")
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"Total duration: {total_duration:.1f} seconds")
    print(f"Average file size: {total_size / total_files / 1024:.1f} KB")
    print(f"Average duration: {total_duration / total_files:.1f} seconds")

if __name__ == "__main__":
    analyze_audio_files()
```

### Batch Audio Conversion
```python
# Convert WAV to MP3 for smaller files
import subprocess
from pathlib import Path

def convert_to_mp3():
    for wav_file in Path('.').glob('tts_*.wav'):
        mp3_file = wav_file.with_suffix('.mp3')
        subprocess.run([
            'ffmpeg', '-i', str(wav_file), 
            '-codec:a', 'libmp3lame', 
            '-b:a', '128k', 
            str(mp3_file)
        ])
        print(f"Converted: {wav_file} -> {mp3_file}")
```

## Troubleshooting

### Common Issues

1. **Audio Not Playing**:
   ```python
   # Test audio file
   import pygame
   pygame.mixer.init()
   pygame.mixer.music.load('tts_20240115_143025.wav')
   pygame.mixer.music.play()
   ```

2. **File Corruption**:
   ```python
   # Validate WAV file
   import wave
   try:
       with wave.open('tts_20240115_143025.wav', 'rb') as w:
           print(f"Channels: {w.getnchannels()}")
           print(f"Sample rate: {w.getframerate()}")
           print(f"Duration: {w.getnframes() / w.getframerate():.2f}s")
   except Exception as e:
       print(f"Corrupted file: {e}")
   ```

3. **Disk Space Issues**:
   ```bash
   # Check disk usage
   du -sh .
   
   # Find largest files
   ls -lah | sort -k5 -hr | head -10
   ```

4. **Permission Problems**:
   ```bash
   # Fix permissions
   chmod 644 *.wav
   chown $USER:$USER *.wav
   ```

### Audio Quality Issues

1. **Robotic Voice**:
   - Try different TTS backend
   - Adjust speech rate (slower = more natural)
   - Use higher quality model

2. **Low Volume**:
   ```python
   # Increase volume in TTS engine
   tts.set_volume(1.0)  # Maximum volume
   ```

3. **Distorted Audio**:
   - Check sample rate compatibility
   - Reduce speech rate
   - Use different audio format

## Integration with External Tools

### Audio Players
```bash
# Play with system default
start tts_20240115_143025.wav  # Windows
open tts_20240115_143025.wav   # Mac
xdg-open tts_20240115_143025.wav  # Linux

# Play with specific players
vlc tts_20240115_143025.wav
mpv tts_20240115_143025.wav
```

### Audio Editing
```bash
# Edit with Audacity
audacity tts_20240115_143025.wav

# Batch processing with SoX
sox tts_20240115_143025.wav output.wav vol 0.5  # Reduce volume
sox tts_20240115_143025.wav output.wav speed 1.2  # Increase speed
```

### Streaming Integration
```python
# Stream audio over network
import socket

def stream_audio(filename, host='localhost', port=8080):
    with open(filename, 'rb') as f:
        data = f.read()
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    sock.sendall(data)
    sock.close()
```

## Performance Optimization

### Reduce File Sizes
```python
# Use lower sample rate for smaller files
TTSEngine(sample_rate=16000)  # Instead of 22050

# Use MP3 compression
TTSEngine(output_format='mp3', bitrate=128)
```

### Faster Generation
```python
# Use fastest TTS backend
TTSEngine(backend='pyttsx3', speech_rate=200)

# Preload models
tts = TTSEngine()
tts.preload_models()  # Load models once at startup
```

---

**Note**: This directory will be automatically populated when the system generates speech. Audio files are safe to delete if you need to free up space - they can always be regenerated from the text logs.