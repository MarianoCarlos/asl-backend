import uvicorn
from fastapi import FastAPI, WebSocket, UploadFile, File
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp

# Paths
MODEL_PATH = "asl_mlp_best.h5"
CSV_PATH = "asl_landmarks.csv"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load labels
labels = pd.read_csv(CSV_PATH)["label"].unique()
labels = sorted(labels)
label_encoder = LabelEncoder().fit(labels)

# Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# FastAPI app
app = FastAPI(title="ASL Translator Backend")

# ✅ CORS setup
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

# --- Image-based prediction ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        img_bytes = await file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Process with Mediapipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return {"prediction": None, "confidence": 0.0}

        # Get landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

        # Normalize
        row = normalize_landmarks(landmarks)

        # Predict
        probs = model.predict(np.array([row]), verbose=0)[0]
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        sign = label_encoder.inverse_transform([class_id])[0]

        return {"prediction": sign, "confidence": confidence}

    except Exception as e:
        return {"error": str(e)}

# --- WebSocket version (image streaming) ---
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()

            np_arr = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                landmarks = [[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]
                row = normalize_landmarks(landmarks)

                probs = model.predict(np.array([row]), verbose=0)[0]
                class_id = int(np.argmax(probs))
                confidence = float(probs[class_id])
                sign = label_encoder.inverse_transform([class_id])[0]

                await websocket.send_json({
                    "prediction": sign,
                    "confidence": confidence
                })
            else:
                await websocket.send_json({"prediction": None, "confidence": 0.0})

    except Exception:
        await websocket.close()
