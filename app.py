import uvicorn
from fastapi import FastAPI, WebSocket
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from fastapi.middleware.cors import CORSMiddleware

# Paths
MODEL_PATH = "asl_mlp_best.h5"
CSV_PATH = "asl_landmarks.csv"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load labels
labels = pd.read_csv(CSV_PATH)["label"].unique()
labels = sorted(labels)
label_encoder = LabelEncoder().fit(labels)

# FastAPI app
app = FastAPI(title="ASL Translator Backend")

# ✅ CORS setup (allow frontend + localhost)
origins = [
    "https://insync-omega.vercel.app",  # Vercel frontend
    "http://localhost:3000"             # local dev
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Normalization ---
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    base = landmarks[0]  # wrist
    landmarks -= base
    max_val = np.max(np.linalg.norm(landmarks, axis=1))
    landmarks /= max_val if max_val > 0 else 1
    return landmarks.flatten()

@app.get("/")
async def root():
    return {"message": "✅ ASL Translator Backend is running!"}

@app.post("/predict")
async def predict(landmarks: dict):
    try:
        row = normalize_landmarks(landmarks["landmarks"])
        probs = model.predict(np.array([row]), verbose=0)[0]
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        sign = label_encoder.inverse_transform([class_id])[0]
        return {"prediction": sign, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            row = normalize_landmarks(data["landmarks"])
            probs = model.predict(np.array([row]), verbose=0)[0]
            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])
            sign = label_encoder.inverse_transform([class_id])[0]

            await websocket.send_json({
                "prediction": sign,
                "confidence": confidence
            })
    except Exception:
        await websocket.close()
