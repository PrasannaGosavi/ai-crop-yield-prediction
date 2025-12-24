from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import requests
import os

# ----------------------------
# App setup
# ----------------------------
app = Flask(__name__)

# 🔥 FORCE CORS (THIS IS THE KEY FIX)
CORS(app, supports_credentials=True)

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

IMAGE_MODEL_PATH = os.path.join(MODEL_DIR, "crop_health_model.h5")
YIELD_MODEL_PATH = os.path.join(MODEL_DIR, "yield_model.pkl")

# ----------------------------
# Load models (compile=False avoids Keras errors)
# ----------------------------
img_model = tf.keras.models.load_model(
    IMAGE_MODEL_PATH,
    compile=False
)

yield_model = pickle.load(open(YIELD_MODEL_PATH, "rb"))

# ----------------------------
# Soil mapping
# ----------------------------
SOIL_MAP = {
    "Black": 1,
    "Red": 2,
    "Alluvial": 3
}

# ----------------------------
# Maharashtra boundary check
# ----------------------------
def is_maharashtra(lat, lon):
    return 15.6 <= lat <= 22.1 and 72.6 <= lon <= 80.9

# ----------------------------
# Weather API
# ----------------------------
def get_weather(lat, lon):
    API_KEY = "Enter Your API Key here"

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    )

    response = requests.get(url).json()

    # 🔒 SAFE CHECK
    if "main" not in response:
        print("Weather API error:", response)
        return 25, 0   # fallback values

    temperature = response["main"]["temp"]
    rainfall = response.get("rain", {}).get("1h", 0)

    return temperature, rainfall


# ----------------------------
# TEST ROUTE (CONFIRM BACKEND)
# ----------------------------
@app.route("/test", methods=["GET"])
@cross_origin()
def test():
    return jsonify({"status": "backend working"})

# ----------------------------
# PREDICTION ROUTE
# ----------------------------
@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():

    try:
        lat = float(request.form["lat"])
        lon = float(request.form["lon"])

        if not is_maharashtra(lat, lon):
            return jsonify({"error": "Service available only in Maharashtra"}), 400

        # Weather
        temperature, rainfall = get_weather(lat, lon)

        # Image
        image = Image.open(request.files["image"]).convert("RGB").resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        health_score = float(img_model.predict(img_array)[0][0]) * 100

        # Soil
        soil = SOIL_MAP[request.form["soil"]]

        # Yield prediction
        predicted_yield = yield_model.predict([
            [health_score, rainfall, temperature, soil]
        ])[0]

        return jsonify({
            "state": "Maharashtra",
            "temperature": round(temperature, 2),
            "rainfall": round(rainfall, 2),
            "health_score": round(health_score, 2),
            "predicted_yield": round(predicted_yield, 2),
            "unit": "kg/hectare"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
