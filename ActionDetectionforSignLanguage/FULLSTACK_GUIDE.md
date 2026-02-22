# SignSense — Full-Stack Sign Language Interpreter

## Project Structure

```
ActionDetectionforSignLanguage/
├── backend/
│   ├── main.py              ← FastAPI server (REST + WebSocket)
│   ├── model_handler.py     ← LSTM model loader/inference
│   └── requirements.txt     ← Python dependencies
├── frontend/
│   ├── index.html           ← Main UI (glassmorphism dark theme)
│   ├── css/
│   │   └── styles.css       ← All styles
│   └── js/
│       └── app.js           ← WebSocket client + UI controller
└── ActionDetectionforSignLanguage/
    └── Action Detection Refined.ipynb  ← Original training notebook
```

---

## Quick Start

### 1. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Make sure `action.h5` is trained and available

Run your notebook to train the model first.
Place `action.h5` inside the `backend/` folder.

### 3. Start the backend

```bash
cd backend
python main.py
# OR
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: http://localhost:8000/docs

### 4. Open the frontend

Open `frontend/index.html` directly in your browser (no build needed).

---

## Backend API Reference

| Method    | Endpoint                          | Description                        |
|-----------|-----------------------------------|------------------------------------|
| GET       | `/api/health`                     | Health check                       |
| GET       | `/api/model/info`                 | Model info (loaded, actions, etc.) |
| GET       | `/api/gestures`                   | All supported gestures             |
| GET       | `/api/session/{id}/stats`         | Session statistics                 |
| GET       | `/api/session/{id}/history`       | Prediction history                 |
| DELETE    | `/api/session/{id}`               | Close a session                    |
| POST      | `/api/session/{id}/reset-sentence`| Clear the sentence buffer          |
| WebSocket | `/ws/{session_id}`                | Real-time video processing         |

### WebSocket Message Format

**Client → Server (send frame):**
```json
{ "type": "frame", "data": "<base64-jpeg>" }
```

**Server → Client (prediction result):**
```json
{
  "type": "prediction",
  "action": "hello",
  "confidence": 0.92,
  "probabilities": { "hello": 0.92, "thanks": 0.03, ... },
  "sentence": ["hello", "thanks"],
  "fps": 24.0,
  "frame_count": 150,
  "landmarks_detected": true,
  "annotated_frame": "<base64-jpeg>",
  "timestamp": "2025-01-01T12:00:00"
}
```

---

## Frontend Features

| Feature | Description |
|---------|-------------|
| Live camera feed | WebRTC webcam capture |
| Annotated feed | Shows hand landmark overlays from backend |
| Sequence buffer bar | Shows how full the 30-frame buffer is |
| Confidence bars | Real-time probability per gesture |
| Prediction hero | Large current gesture with confidence ring |
| Sentence builder | Accumulates gestures into a sentence |
| Text-to-Speech | Browser Web Speech API, toggle on/off |
| Copy sentence | Copy full translated sentence to clipboard |
| History log | Last 20 detected gestures |
| Usage chart | Session-level gesture frequency bars |
| Session stats | Total predictions, avg confidence, uptime, frames |
| Gesture dictionary | Visual reference card for all gestures |
| Settings panel | Backend URL, threshold, FPS, TTS, landmarks |
| Keyboard shortcuts | Space=pause, F=flip, Ctrl+C=copy, Esc=close modal |
| Dark glassmorphism UI | Professional animated dark theme |
| Toast notifications | Non-intrusive status messages |

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Pause / Resume detection |
| `F` | Flip camera |
| `Ctrl+C` | Copy translated sentence |
| `Esc` | Close settings panel |

---

## What You Can Add (Suggestions)

### Frontend Additions

| Feature | Difficulty | Description |
|---------|-----------|-------------|
| **Dark/Light theme toggle** | Easy | Add CSS variable swap |
| **Multi-language TTS** | Easy | Change SpeechSynthesis voice locale |
| **Sentence export (PDF/TXT)** | Easy | Add download button |
| **Sound feedback on detection** | Easy | Play a beep on new gesture |
| **Gesture learning mode** | Medium | Show how-to animation per gesture |
| **MediaPipe in-browser** | Medium | Run MediaPipe.js directly in browser (no backend needed) |
| **Session recording** | Medium | Record webcam + detections to video |
| **ASL alphabet mode** | Medium | Add A-Z letter detection UI |
| **Dark mode overlay on camera** | Easy | CSS filter tint on video |
| **Accessibility mode** | Medium | High-contrast + larger fonts |
| **Haptic feedback** | Easy | `navigator.vibrate()` on mobile |
| **Progressive Web App (PWA)** | Medium | Offline support, install to home screen |
| **User login + saved sessions** | Hard | Auth + session history persistence |
| **Real-time chart (Chart.js)** | Medium | Live confidence line chart |
| **Emotion overlay** | Hard | Add face expression detection |

### Backend Additions

| Feature | Difficulty | Description |
|---------|-----------|-------------|
| **More gestures (ASL A-Z)** | Hard | Retrain with alphabet dataset |
| **Indian Sign Language (ISL) support** | Hard | ISL-specific training dataset |
| **Model versioning** | Medium | Save/load multiple model versions |
| **Training API endpoint** | Hard | `POST /api/train` triggers retraining |
| **Data collection endpoint** | Medium | `POST /api/collect` saves new training samples |
| **Model accuracy metrics** | Medium | Return confusion matrix, per-class accuracy |
| **Two-hand gesture support** | Medium | Use both hands independently |
| **Phrase/sentence database** | Medium | Map gesture sequences to full phrases |
| **Redis session storage** | Medium | Persist sessions across restarts |
| **PostgreSQL logging** | Medium | Store prediction history in database |
| **User authentication** | Medium | JWT-based auth for multi-user |
| **REST API rate limiting** | Easy | `slowapi` library |
| **Docker containerization** | Easy | Dockerfile + docker-compose |
| **GPU acceleration** | Medium | TF GPU build for faster inference |
| **Confidence calibration** | Medium | Temperature scaling on softmax output |
| **Sequence smoothing** | Medium | Kalman filter on predictions |
| **WebRTC server-side** | Hard | Replace base64 stream with RTSP/WebRTC |
| **Mobile API (FastAPI + Expo)** | Hard | React Native app connecting to same backend |
| **Webhook notifications** | Easy | POST to external URL on gesture detection |

---

## Model Improvements

| Improvement | Why |
|-------------|-----|
| **Add more training sequences** | Current ~24 sequences/action → need 100+ |
| **Data augmentation** | Mirror, rotate, add noise to sequences |
| **Transformer model** | Replace LSTM with vision transformer |
| **Transfer learning** | Fine-tune on MediaPipe foundation models |
| **Bidirectional LSTM** | Better context for gesture boundaries |
| **Attention mechanism** | Focus on most informative frames |
| **Include face landmarks** | Better distinguish similar gestures |
| **Confidence calibration** | Fix overconfident predictions |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5 + CSS3 + Vanilla JS |
| Backend | Python 3.10 + FastAPI |
| Real-time | WebSocket (native browser + websockets lib) |
| ML | TensorFlow / Keras |
| Vision | MediaPipe Holistic |
| Video | OpenCV + WebRTC (browser) |
| TTS | Web Speech API (browser) |
