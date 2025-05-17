from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model only once when starting the server
model = tf.keras.models.load_model("animal_classifier.h5")

# Class names (replace with your actual classes)
CLASS_NAMES = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "spider", "squirrel"]

def preprocess_image(image):
    """Helper function to preprocess the image"""
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file = request.files["file"].read()
        image = Image.open(io.BytesIO(file))
        
        # Validate image
        if image.format not in ['JPEG', 'PNG', 'JPG']:
            return jsonify({"error": "Invalid image format. Please upload JPEG or PNG"}), 400
            
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return jsonify({
            "class": predicted_class,
            "confidence": confidence
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)