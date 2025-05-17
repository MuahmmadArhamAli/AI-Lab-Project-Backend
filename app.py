from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import os
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMAGE_SIZE = (224, 224)

# Load model
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "animal10n_classifier.keras"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "spider", "squirrel"]

def is_allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Helper function to preprocess the image"""
    # Convert to RGB if image is RGBA
    if image.mode in ('RGBA', 'LA'):
        image = image.convert('RGB')
    
    image = image.resize(IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    # Check for empty filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Check file extension
    if not is_allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_length = file.tell()
    file.seek(0)  # Reset file pointer
    if file_length > MAX_FILE_SIZE:
        return jsonify({"error": f"File too large. Max size: {MAX_FILE_SIZE//1024//1024}MB"}), 400
    
    try:
        # Verify and process image
        image = Image.open(file.stream)
        
        # Additional verification
        image.verify()  # Verify that it is, in fact, an image
        file.stream.seek(0)  # Reset stream position
        image = Image.open(file.stream)  # Reopen after verify
        
        # Process the image
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        
        # Get results
        predicted_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(np.max(predictions[0]))
        
        return jsonify({
            "class": predicted_class,
            "confidence": confidence,
            "all_predictions": {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
        })
        
    except UnidentifiedImageError:
        return jsonify({"error": "Cannot identify image file"}), 400
    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "allowed_extensions": list(ALLOWED_EXTENSIONS),
        "max_file_size": MAX_FILE_SIZE
    }), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)