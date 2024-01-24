from flask import Flask
from flask import jsonify
from flask import request

import tensorflow as tf
import cv2
import numpy as np
from flask import render_template

app = Flask(__name__)

#the model that will be used for prediction
fer = tf.keras.models.load_model("model2_epoch25.h5")

@app.route("/predictFacialEmotion", methods=["POST"])
def predictFacialEmotion():
    try:
        filestream = request.files["file"].read()

        imgbytes = np.fromstring(filestream, np.uint8)
        img = cv2.imdecode(imgbytes, cv2.IMREAD_COLOR)

        #processing the img
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale(gray_img, 1.1, 4)

        for x, y, w, h in faces:
            roi_gray = gray_img[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            faces_in_roi = faceCascade.detectMultiScale(roi_gray)

            if len(faces_in_roi) == 0:
                print("Face not detected in ROI")
            else:
                for (ex, ey, ew, eh) in faces_in_roi:
                    face_roi = roi_color[ey: ey+eh, ex:ex + ew]

        final_image = cv2.resize(face_roi, (224, 224))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image / 255.0  # Normalizing

        prediction = fer.predict(final_image)

        emotion_predicted = ""

        print(prediction[0])

        if (np.argmax(prediction) == 0):
            emotion_predicted = "Angry"
            
        elif (np.argmax(prediction) == 1):
            emotion_predicted = "Disgust"
            
        elif (np.argmax(prediction) == 2):
            emotion_predicted = "Fear"
            
        elif (np.argmax(prediction) == 3):
            emotion_predicted = "Happy"
            
        elif (np.argmax(prediction) == 4):
            emotion_predicted = "Sad"
            
        elif (np.argmax(prediction) == 5):
            emotion_predicted = "Surprised"
            
        elif (np.argmax(prediction) == 6):
            emotion_predicted = "Neutral"
        
        return jsonify(emotion_predicted)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)