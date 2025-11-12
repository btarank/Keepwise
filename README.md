# Keepwise

**A meeting audio recording, transcription and sentiment / diarization tool.**  
This repo contains the Keepwise Flask app used for meeting recording, ASR (Vosk), sentiment analysis, diarization utilities, and a small web UI for uploading/processing audio.

## Features
- Record or upload meeting audio (mp3/wav)
- Offline speech-to-text with Vosk
- Speaker diarization & audio split helpers
- Sentiment analysis + risk heuristics
- Demo UI for single file and bulk processing

## Tech stack
- Python 3.9+
- Flask
- Vosk (offline ASR)
- SQLite (simple local storage)
- Additional libs: soundfile, joblib, pandas, etc.

## Quick setup (Windows, Git Bash)
1. Clone (you already have this):
```bash
git clone https://github.com/btarank/Keepwise.git
cd Keepwise

