from flask import Flask, request, jsonify
import numpy as np
import cv2
import mediapipe as mp
import pickle
import tensorflow as tf

# ------------------------------
# Load Model + Label Encoder
# ------------------------------
model = tf.keras.models.load_model("linux.h5")

with open("krishnav.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ------------------------------
# Mediapipe Initialization
# ------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# ------------------------------
# Preprocessing Functions
# ------------------------------
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 2)
    base = landmarks[0]
    centered = landmarks - base
    max_value = np.max(np.linalg.norm(centered, axis=1))
    return (centered / max_value).reshape(42, 2, 1) if max_value != 0 else centered.reshape(42, 2, 1)

def extract_landmarks(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None

    # If 1 hand detected → pad second hand with zeros
    if len(results.multi_hand_landmarks) == 2:
        l1 = [[lm.x, lm.y] for lm in results.multi_hand_landmarks[0].landmark]
        l2 = [[lm.x, lm.y] for lm in results.multi_hand_landmarks[1].landmark]
    else:
        l1 = [[lm.x, lm.y] for lm in results.multi_hand_landmarks[0].landmark]
        l2 = [[0, 0]] * 21

    combined = l1 + l2  # 42 points
    return normalize_landmarks(combined)

# ------------------------------
# Flask App
# ------------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "ISL Model API is running!"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files['image']
        if file is None:
            return jsonify({"error": "No image uploaded"}), 400

        # Read image
        img_bytes = file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # Extract landmarks
        landmarks = extract_landmarks(img)
        if landmarks is None:
            return jsonify({"error": "Hand not detected"}), 400

        # Model expects shape (1,42,2,1)
        input_data = np.expand_dims(landmarks, axis=0)

        # Predict
        preds = model.predict(input_data)
        class_index = np.argmax(preds)
        class_label = label_encoder.inverse_transform([class_index])[0]

        return jsonify({
            "prediction": class_label,
            "confidence": float(np.max(preds))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "_main_":
    app.run(debug=True)

