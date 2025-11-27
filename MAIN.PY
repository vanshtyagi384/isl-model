from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import pickle
import numpy as np
from pydantic import BaseModel
from typing import List
from googletrans import Translator

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = tf.keras.models.load_model("linux.h5")
with open("krishnav.pkl", "rb") as f:
    label_encoder = pickle.load(f)


def normalize_and_reshape_for_cnn(landmarks):
    landmarks_np = np.array(landmarks).reshape(42, 2)
    base = landmarks_np[0]
    centered = landmarks_np - base
    max_value = np.max(np.linalg.norm(centered, axis=1))

    normalized = centered / max_value if max_value > 0 else centered
    return normalized.reshape(42, 2, 1)


class HandLandmarks(BaseModel):
    landmarks: List[List[List[float]]]


@app.get("/")
def home():
    return {"status": "ISHARA API LIVE"}


@app.post("/predict")
async def predict(data: HandLandmarks):
    try:
        # Extract the 42 landmark points
        landmarks_array = np.array(data.landmarks)[0]

        processed = normalize_and_reshape_for_cnn(landmarks_array)
        processed = np.expand_dims(processed, axis=0)

        prediction = model.predict(processed)
        confidence = np.max(prediction)

        if confidence > 0.9:
            label_index = np.argmax(prediction)
            label = label_encoder.inverse_transform([label_index])[0]
            return {"prediction": label, "confidence": float(confidence)}
        else:
            return {"prediction": "", "confidence": float(confidence)}

    except Exception as e:
        return {"error": str(e)}


class TranslationRequest(BaseModel):
    text: str
    src_lang: str
    dest_lang: str


@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        translator = Translator()
        translated = translator.translate(request.text, src=request.src_lang, dest=request.dest_lang)
        return {"translated_text": translated.text}
    except Exception as e:
        return {"error": str(e)}