import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Cargar modelo y etiquetas
model = load_model('emotion_mobilenet_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Cargar clasificador de rostro
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Captura desde la webcam
cap = cv2.VideoCapture(0)
img_size = 224  # Tamaño de entrada para MobileNetV2

print("Presiona 'q' para salir...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (img_size, img_size))
        face_array = np.expand_dims(face_img, axis=0) / 255.0

        preds = model.predict(face_array)
        emotion = emotion_labels[np.argmax(preds)]

        # Mostrar resultado
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Detección de emociones (presiona q para salir)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
