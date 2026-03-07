## ActionDetectionforSignLanguage — Full-Stack Sign Language Interpreter

ActionDetectionforSignLanguage is a full‑stack sign language interpreter that uses computer vision and deep learning to recognize hand gestures from live video and translate them into text (and optional speech).  
The project combines a Python FastAPI backend with a modern, browser‑based frontend for real‑time interaction.

---

## Project Structure

```text
ActionDetectionforSignLanguage/
├── ActionDetectionforSignLanguage/
│   └── Action Detection Refined.ipynb   ← Original training notebook
├── backend/
│   ├── main.py                          ← FastAPI server (REST + WebSocket)
│   ├── model_handler.py                 ← LSTM model loader/inference
│   ├── seed_dataset.py                  ← Optional dataset seeding script
│   └── requirements.txt                 ← Python dependencies
├── frontend/
│   ├── index.html                       ← Main UI
│   ├── css/
│   │   └── styles.css                   ← Styles (dark glassmorphism theme)
│   └── js/
│       └── app.js                       ← WebSocket client + UI controller
└── README.md                            ← You are here
```

For a more detailed breakdown of features and future improvements, see `ActionDetectionforSignLanguage/FULLSTACK_GUIDE.md`.

---

## Features

- **Real-time sign language gesture detection** from webcam input
- **LSTM-based action recognition model** running in the backend
- **WebSocket streaming** for low‑latency frame processing and predictions
- **Live UI visualizations**: confidence bars, sentence builder, history, stats
- **Text-to-speech output** (browser Web Speech API)
- **Dark, accessible UI** designed for clear, low‑distraction usage

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/mohitkumar402/ActionDetectionforSignLanguage.git
cd ActionDetectionforSignLanguage
```

### 2. Set up the backend (Python)

It is recommended to use a virtual environment.

```bash
cd ActionDetectionforSignLanguage/backend
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Provide the trained model (`action.h5`)

1. Train your model using the notebook in `ActionDetectionforSignLanguage/Action Detection Refined.ipynb`,  
   **or** use your own trained `action.h5`.
2. Place the resulting `action.h5` file inside the `backend/` folder.

Dataset notes:
- You can create your own dataset (e.g., gestures like *hello*, *hi*, *I love you*, *thanks*, etc.).
- You can also adapt public sign language datasets (e.g., RWTH‑PHOENIX‑Weather, ASLLVD) to this format.

### 4. Run the backend server

From the `backend/` directory:

```bash
python main.py
# or, using uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open API docs at: `http://localhost:8000/docs`

### 5. Open the frontend

From the repository root:

```bash
cd ActionDetectionforSignLanguage/frontend
```

Then open `index.html` in your browser:

- Double‑click `index.html`, or
- Serve it with any simple static server (e.g., Live Server in VS Code).

The frontend will connect to the backend WebSocket (by default `ws://localhost:8000/ws/{session_id}`) and start streaming frames when you enable the camera.

---

## Backend API (Overview)

Key endpoints (see `/docs` for full details):

- `GET /api/health` — Health check
- `GET /api/model/info` — Model status and metadata
- `GET /api/gestures` — List of supported gestures
- `GET /api/session/{id}/stats` — Session statistics
- `GET /api/session/{id}/history` — Recent predictions
- `DELETE /api/session/{id}` — Close a session
- `POST /api/session/{id}/reset-sentence` — Clear the current sentence
- `WebSocket /ws/{session_id}` — Real‑time frame streaming and predictions

WebSocket messages follow a simple JSON protocol (frames from client, predictions from server); see `FULLSTACK_GUIDE.md` for exact formats.

---

## Model & Training

The core model is an action recognition network (e.g., CNN + LSTM) trained on sequences of pose/keypoint features extracted from sign language video.

Typical training pipeline:

1. **Collect / prepare dataset** of labeled sign gestures.
2. **Extract keypoints** using MediaPipe (hands, pose, face as needed).
3. **Train the sequence model** (LSTM or Transformer) on these keypoints.
4. **Export** the trained model as `action.h5` and place it in `backend/`.

You can customize:

- Number and type of gestures (ASL, ISL, custom signs, etc.)
- Model architecture (LSTM, bi‑LSTM, Transformer)
- Input features (single hand vs. both hands, inclusion of face/pose)

---

## Contributing

Contributions are welcome!  
You can:

- Open issues for bugs, ideas, or improvements.
- Submit pull requests for:
  - New gestures or datasets
  - UI/UX improvements in the frontend
  - Backend API enhancements or performance optimizations
  - Better documentation and examples

Please keep changes small and focused where possible.

---

## License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.

---

## Acknowledgments

- Open‑source sign language datasets and research papers on sign/action recognition
- MediaPipe, TensorFlow/Keras, FastAPI, and the broader open‑source community

Let’s continue using AI to help **bridge the communication gap** for the Deaf and hard‑of‑hearing community.
