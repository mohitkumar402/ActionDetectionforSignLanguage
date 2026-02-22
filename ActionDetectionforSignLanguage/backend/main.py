"""
Sign Language Action Detection - FastAPI Backend
Provides REST API + WebSocket for real-time gesture recognition.
Optimized: pre-allocated buffers, async decode, reduced payload, cached zeros.
"""

import asyncio
import base64
import json
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Try to import model handler; graceful degradation if model not trained yet
try:
    from model_handler import GestureModel
    MODEL_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Model not available: {e}")
    MODEL_AVAILABLE = False

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sign Language Detection API",
    description="Real-time sign language gesture recognition using LSTM + MediaPipe",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend from /static
try:
    app.mount("/static", StaticFiles(directory="../frontend"), name="static")
except Exception:
    pass

# ─── MediaPipe Setup ───────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ─── Global State ─────────────────────────────────────────────────────────────
ACTIONS = np.array(["hello", "thanks", "iloveyou", "one", "two", "three"])
SEQUENCE_LENGTH = 30
THRESHOLD = 0.5

# Pre-allocated zero arrays to avoid per-frame allocation
_ZERO_HAND = np.zeros(63, dtype=np.float32)          # 21 landmarks × 3
_JPEG_ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 75]  # quality 75 = fast + acceptable size

gesture_model: Optional["GestureModel"] = None
active_sessions: Dict[str, dict] = {}

# Per-session stats
session_stats: Dict[str, dict] = {}

# ─── Pydantic Models ──────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    action: str
    confidence: float
    probabilities: Dict[str, float]
    sentence: List[str]
    fps: float
    frame_count: int
    timestamp: str

class SessionStats(BaseModel):
    session_id: str
    total_predictions: int
    gesture_counts: Dict[str, int]
    avg_confidence: float
    uptime_seconds: float

class ModelInfo(BaseModel):
    loaded: bool
    actions: List[str]
    sequence_length: int
    threshold: float
    model_path: Optional[str]

class GestureHistoryItem(BaseModel):
    gesture: str
    confidence: float
    timestamp: str

# ─── Helpers ──────────────────────────────────────────────────────────────────
def extract_keypoints(results) -> np.ndarray:
    """Extract left + right hand keypoints. Uses pre-allocated zeros to avoid per-frame alloc."""
    lh = (np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark], dtype=np.float32).ravel()
          if results.left_hand_landmarks else _ZERO_HAND)
    rh = (np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark], dtype=np.float32).ravel()
          if results.right_hand_landmarks else _ZERO_HAND)
    return np.concatenate([lh, rh])


def decode_frame(b64_frame: str) -> Optional[np.ndarray]:
    """Decode base64 JPEG frame to numpy array."""
    try:
        img_bytes = base64.b64decode(b64_frame, validate=False)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def encode_frame(frame: np.ndarray) -> str:
    """Encode numpy frame to base64 JPEG. Quality 75 for speed."""
    _, buffer = cv2.imencode(".jpg", frame, _JPEG_ENCODE_PARAMS)
    return base64.b64encode(buffer).decode("ascii")


def draw_landmarks_on_frame(frame: np.ndarray, results) -> np.ndarray:
    """Draw hand landmarks on frame with custom styles."""
    annotated = frame.copy()

    # Right hand - cyan
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            results.right_hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 200, 200), thickness=2),
        )

    # Left hand - magenta
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            results.left_hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(200, 0, 200), thickness=2),
        )

    return annotated


def get_or_create_session(session_id: str) -> dict:
    if session_id not in active_sessions:
        active_sessions[session_id] = {
            "sequence": deque(maxlen=SEQUENCE_LENGTH),
            "sentence": [],
            "predictions": deque(maxlen=10),
            "frame_count": 0,
            "start_time": time.time(),
            "last_fps_time": time.time(),
            "fps_frame_count": 0,
            "current_fps": 0.0,
            "history": [],
            "gesture_counts": {a: 0 for a in ACTIONS},
            "total_confidence_sum": 0.0,
            "total_predictions": 0,
        }
    return active_sessions[session_id]


# ─── REST Endpoints ───────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Sign Language Detection API is running", "docs": "/docs"}


@app.get("/api/model/info", response_model=ModelInfo)
async def model_info():
    return ModelInfo(
        loaded=MODEL_AVAILABLE and gesture_model is not None,
        actions=ACTIONS.tolist(),
        sequence_length=SEQUENCE_LENGTH,
        threshold=THRESHOLD,
        model_path="action.h5" if MODEL_AVAILABLE else None,
    )


@app.get("/api/gestures")
async def get_gestures():
    """Return all supported gesture labels with metadata."""
    gestures = []
    for i, action in enumerate(ACTIONS):
        gestures.append({
            "id": i,
            "name": action,
            "label": action.upper(),
            "emoji": get_gesture_emoji(action),
            "description": get_gesture_description(action),
        })
    return {"gestures": gestures, "count": len(gestures)}


@app.get("/api/session/{session_id}/stats", response_model=SessionStats)
async def get_session_stats(session_id: str):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    uptime = time.time() - session["start_time"]
    avg_conf = (session["total_confidence_sum"] / session["total_predictions"]) \
        if session["total_predictions"] > 0 else 0.0
    return SessionStats(
        session_id=session_id,
        total_predictions=session["total_predictions"],
        gesture_counts=session["gesture_counts"],
        avg_confidence=round(avg_conf, 4),
        uptime_seconds=round(uptime, 2),
    )


@app.get("/api/session/{session_id}/history")
async def get_session_history(session_id: str, limit: int = 50):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"history": session["history"][-limit:]}


@app.delete("/api/session/{session_id}")
async def close_session(session_id: str):
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"message": f"Session {session_id} closed"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.post("/api/session/{session_id}/reset-sentence")
async def reset_sentence(session_id: str):
    session = get_or_create_session(session_id)
    session["sentence"] = []
    return {"message": "Sentence reset", "sentence": []}


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL_AVAILABLE and gesture_model is not None,
        "active_sessions": len(active_sessions),
        "timestamp": datetime.now().isoformat(),
    }


# ─── WebSocket ────────────────────────────────────────────────────────────────
# New architecture: browser runs MediaPipe → sends 126 keypoint floats
# Server only runs LSTM inference → ~10x faster, no frame encode/decode
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = get_or_create_session(session_id)

    _empty_probs = {a: 0.0 for a in ACTIONS}
    print(f"[WS] Session connected: {session_id}")

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            # ── KEYPOINTS path (new fast architecture) ──────────────────────
            if msg.get("type") == "keypoints":
                kp_data = msg.get("data", [])
                if len(kp_data) != 126:
                    continue

                keypoints = np.array(kp_data, dtype=np.float32)
                session["sequence"].append(keypoints)
                session["frame_count"] += 1

                # FPS
                session["fps_frame_count"] += 1
                now = time.time()
                elapsed = now - session["last_fps_time"]
                if elapsed >= 1.0:
                    session["current_fps"] = round(session["fps_frame_count"] / elapsed, 1)
                    session["fps_frame_count"] = 0
                    session["last_fps_time"] = now

                response = {
                    "type": "prediction",
                    "action": "",
                    "confidence": 0.0,
                    "probabilities": _empty_probs,
                    "sentence": session["sentence"],
                    "fps": session["current_fps"],
                    "frame_count": session["frame_count"],
                    "buffer": min(len(session["sequence"]), SEQUENCE_LENGTH),
                    "timestamp": datetime.now().isoformat(),
                }

                # LSTM prediction — only when sequence buffer is full
                if (
                    gesture_model is not None
                    and gesture_model.is_loaded
                    and len(session["sequence"]) == SEQUENCE_LENGTH
                ):
                    seq_array = np.expand_dims(
                        np.array(session["sequence"], dtype=np.float32), axis=0
                    )
                    raw_probs = gesture_model.predict(seq_array)[0]

                    best_idx = int(np.argmax(raw_probs))
                    session["predictions"].append(best_idx)

                    # Majority vote smoothing over last 10 predictions
                    unique, counts = np.unique(list(session["predictions"]), return_counts=True)
                    smoothed_idx   = int(unique[np.argmax(counts)])
                    smoothed_action = ACTIONS[smoothed_idx]
                    smoothed_conf   = float(raw_probs[smoothed_idx])

                    if smoothed_conf > THRESHOLD:
                        if not session["sentence"] or session["sentence"][-1] != smoothed_action:
                            session["sentence"].append(smoothed_action)
                            if len(session["sentence"]) > 10:
                                session["sentence"] = session["sentence"][-10:]
                            session["history"].append({
                                "gesture": smoothed_action,
                                "confidence": round(smoothed_conf, 4),
                                "timestamp": datetime.now().isoformat(),
                            })
                            session["gesture_counts"][smoothed_action] = \
                                session["gesture_counts"].get(smoothed_action, 0) + 1
                            session["total_predictions"] += 1
                            session["total_confidence_sum"] += smoothed_conf

                    response["action"]        = smoothed_action
                    response["confidence"]    = round(smoothed_conf, 4)
                    response["probabilities"] = {
                        ACTIONS[i]: round(float(raw_probs[i]), 4) for i in range(len(ACTIONS))
                    }
                    response["sentence"] = session["sentence"]

                await websocket.send_text(json.dumps(response))

            # ── Control messages ────────────────────────────────────────────
            elif msg.get("type") == "reset_sentence":
                session["sentence"] = []
                await websocket.send_text(json.dumps({"type": "sentence_reset", "sentence": []}))

            elif msg.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        print(f"[WS] Session disconnected: {session_id}")
    except Exception as e:
        print(f"[WS] Error in session {session_id}: {e}")
    finally:
        if session_id in active_sessions:
            del active_sessions[session_id]


# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global gesture_model
    if MODEL_AVAILABLE:
        try:
            gesture_model = GestureModel()
            gesture_model.load("action.h5")
            print("[OK] Model loaded: action.h5")
        except FileNotFoundError:
            print("[WARN] action.h5 not found. Train the model first.")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
    print("[OK] Sign Language Detection API started")
    print("[OK] Docs available at: http://localhost:8000/docs")


# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_gesture_emoji(action: str) -> str:
    emojis = {
        "hello": "👋",
        "thanks": "🙏",
        "iloveyou": "🤟",
        "one": "☝️",
        "two": "✌️",
        "three": "🤙",
    }
    return emojis.get(action.lower(), "🤚")


def get_gesture_description(action: str) -> str:
    descriptions = {
        "hello": "Wave your hand as a greeting gesture",
        "thanks": "Press both palms together in a thankful gesture",
        "iloveyou": "Extend thumb, index, and pinky fingers",
        "one": "Raise your index finger",
        "two": "Raise index and middle fingers in a V shape",
        "three": "Raise three fingers",
    }
    return descriptions.get(action.lower(), "Perform the gesture in front of camera")


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
