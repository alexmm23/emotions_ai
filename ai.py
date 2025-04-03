import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Cargar el modelo entrenado
model = tf.keras.models.load_model('modelo_facial.h5')

# Inicializar MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Función para preprocesar la imagen para la red neuronal
def preprocess_face(face, size=(48, 48)):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)   # Escala de grises
    face = cv2.resize(face, size)                   # Redimensionar
    face = face / 255.0                             # Normalizar
    face = np.expand_dims(face, axis=0)             # Añadir dimensión batch
    face = np.reshape(face, (1, 48, 48, 1))
    return face

# Etiquetas de emociones
emotions = ['Enojado', 'Asco', 'Miedo', 'Feliz', 'Triste', 'Sorpresa', 'Neutral']

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a RGB para MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Obtener coordenadas del rectángulo de la cara
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Extraer la cara detectada
            face = frame[y_min:y_max, x_min:x_max]

            if face.shape[0] > 0 and face.shape[1] > 0:
                face = preprocess_face(face)
                
                # Usar datos de sensores simulados o valores fijos
                # Valores neutrales que no afecten mucho la predicción
                # sensor_data = np.zeros((1, 3))  # Valores neutrales (0,0,0)
                
                # Realizar la predicción con ambas entradas
                prediction = model.predict(face, verbose=0)  # Añadir verbose=0 para evitar mensajes
                emotion = emotions[np.argmax(prediction)]

                # Dibujar el rectángulo y la emoción detectada
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector - MediaPipe', frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()