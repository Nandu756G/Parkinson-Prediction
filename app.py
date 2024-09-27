from flask import Flask, request, render_template,jsonify
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model(r'C:\Users\Nandu Gatla\Downloads\parkinsons_dataset project\parkinson.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        image = np.array(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image_resized = cv2.resize(image, (128, 128))
        image_scaled = image_resized / 255.0
        image_reshaped = np.reshape(image_scaled, [1, 128, 128, 3])

        prediction = model.predict(image_reshaped)
        pred_label = np.argmax(prediction)
        result=""
        if pred_label == 0:
            result = 'Normal'
        else:
            result = 'Parkinson\'s disease'
        
        return jsonify({'Prediction':result})
    

app.run()
