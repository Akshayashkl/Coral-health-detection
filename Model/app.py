    
import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter  # Use TensorFlow Lite runtime for loading models
from pymongo import MongoClient
import logging
logging.getLogger("pymongo").setLevel(logging.WARNING)
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Paths to the TensorFlow Lite models
cnn_model_path = "C:\\newprograms\\All_Projects\\Corals_new\\Model\\all_models\\cnn_model_TFL.tflite"
resnet_model_path = "C:\\newprograms\\All_Projects\\Corals_new\Model\\all_models\\resnet50_model_TFL.tflite"
densenet_model_path = "C:\\newprograms\\All_Projects\\Corals_new\\Model\\all_models\\densenet_model_TFL.tflite"
efficientNet_model_path ="C:\\newprograms\\All_Projects\\Corals_new\Model\\all_models\\models\\saved_model\\efficientnet_float32.tflite"

# Function to load TensorFlow Lite models
def load_tflite_model(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Load the TensorFlow Lite models
cnn_model = load_tflite_model(cnn_model_path)
resnet_model = load_tflite_model(resnet_model_path)
densenet_model= load_tflite_model(densenet_model_path)
efficientNet_model= load_tflite_model(efficientNet_model_path)

# Function to preprocess the image
def preprocess_image(image,target_size):
    image = image.resize(target_size)  # Resize to match the input size of the models
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions using TensorFlow Lite models
def predict_with_tflite(interpreter, input_data):
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    # Set the input tensor
    interpreter.set_tensor(input_index, input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_index)
    return output_data
# Replace with your actual connection string
MONGO_URI = "mongodb+srv://coral_health:TSSXDayCih0LKnrk@coral-health.om41rpe.mongodb.net"

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
db = client['coral_predictions']
collection = db['results']


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        logging.error("No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    model_type = request.form.get("model")

    logging.debug(f"Received request with model: {model_type}")

    if file.filename == '':
        logging.error("No file selected")
        return jsonify({'error': 'No file selected or invalid type'}), 400

    if model_type not in ["resnet", "cnn","densenet","efficientnet"]:
        return jsonify({"error": f"Invalid model choice: {model_type}. Choose 'resnet', 'cnn', 'densenet',or 'efficientnet' "}), 400

    try:
        if model_type == "resnet":
            interpreter = resnet_model
            input_size = (128,128)
        elif model_type == "densenet":
            interpreter = densenet_model
            input_size = (128,128)
        elif model_type == "efficientnet":
            interpreter = efficientNet_model
            input_size = (224,224)
        else:
            interpreter = cnn_model
            input_size = (128,128)
         
        image = preprocess_image(Image.open(file.stream),input_size)
        prediction = predict_with_tflite(interpreter, image)

        if model_type == "efficientnet":
            # Softmax output with 2 class probabilities
            # Apply softmax to logits
            logits = prediction[0]
            exp_scores = np.exp(logits - np.max(logits))  # for numerical stability
            probs = exp_scores / np.sum(exp_scores)
            print("Softmax Probabilities:", probs)
            class_index = int(np.argmax(probs))
            confidence = float(np.max(probs))

        else:
            # Sigmoid output
            class_index = 1 if prediction[0][0] >= 0.5 else 0
            confidence = float(prediction[0][0]) if class_index == 1 else 1 - float(prediction[0][0])

        result = "Healthy" if class_index == 1 else "Bleached"
        # Save prediction to MongoDB
        collection.insert_one({
            "model": model_type,
            "result": result,
            "confidence": confidence,
            "filename": file.filename
        })

        return jsonify({
            'result': result,
            'confidence': float(f"{confidence * 100:.4f}")
        })

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    logging.info(f"Starting the server on port {port}...")
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=port)
        logging.info(f"Server is running on http://0.0.0.0:{port}")
    except Exception as e:
        logging.error(f"Failed to start the server: {e}")
        


