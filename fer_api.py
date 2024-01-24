from flask import Flask
from flask import jsonify
from flask import request

import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

#the model that will be used for prediction
fer = tf.keras.models.load_model("model2_epoch25.h5")

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprised", "Neutral"]

@app.route("/predictFacialEmotion", methods=["POST"])
def predict_facial_emotion():
    try:
        # Check if the request contains a file
        if "file" not in request.files:
            return jsonify({"error": "No file provided"})

        file_stream = request.files["file"].read()

        # Decode the image
        img_bytes = np.frombuffer(file_stream, dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        # Preprocess the image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray_img, 1.1, 4)

        if len(faces) == 0:
            return jsonify({"error": "No face detected"})

        x, y, w, h = faces[0]
        face_roi = img[y:y+h, x:x+w]

        final_image = cv2.resize(face_roi, (224, 224))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image / 255.0  # Normalizing

        # Make prediction
        prediction = fer.predict(final_image)

        # Get the predicted emotion
        predicted_emotion_index = np.argmax(prediction)
        emotion_predicted = emotion_labels[predicted_emotion_index]

        return jsonify({"emotion": emotion_predicted, "confidence": float(prediction[0, predicted_emotion_index])})
    
    except Exception as e:
        return jsonify({"error": "Error processing the request", "details": str(e)})

if __name__ == "__main__":
    app.run(debug=True)