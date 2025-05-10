from flask import Flask, request, jsonify
import cv2
import numpy as np
from deepface import DeepFace
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Inicializar la app Flask
app = Flask(__name__)

# Inicializar Firebase
cred = credentials.Certificate("firebase_credentials.json")  # Ruta al archivo JSON de credenciales
firebase_admin.initialize_app(cred)
db = firestore.client()

# Lista de emociones (DeepFace detecta estas emociones por defecto)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Verificar si se envió un archivo
        # if 'image' not in request.files:
        #     return jsonify({'error': 'No se envió ninguna imagen'}), 400

        # # Leer la imagen enviada
        # file = request.files['image']
        
        # Convertir los datos de la solicitud en una imagen
        np_img = np.frombuffer(request.data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Crear el directorio '/img' si no existe
        import os
        os.makedirs('./img', exist_ok=True)

        # Guardar la imagen en el directorio '/img' con un nombre único
        img_path = f'./img/{datetime.now().strftime("%Y%m%d%H%M%S%f")}.jpg'
        cv2.imwrite(img_path, img)

        # Convertir la imagen a escala de grises para detección de rostros
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Cargar el clasificador Haar Cascade para detectar rostros
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return jsonify({'error': 'No se detectaron rostros en la imagen'}), 400

        # Procesar cada rostro detectado
        for (x, y, w, h) in faces:
            # Extraer la región del rostro (ROI)
            face_roi = img[y:y + h, x:x + w]

            # Analizar emociones con DeepFace
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Manejar el caso de múltiples resultados
            if isinstance(result, list):
                result = result[0]  # Tomar el primer resultado si es una lista

            # Obtener la emoción dominante
            emotion = result['dominant_emotion']
            confidence = result['emotion'][emotion]

            # Simular datos de BPM y sudoración (en un caso real, estos vendrían de sensores)
            bpm = np.random.randint(60, 100)  # Simulación de BPM
            sweat_level = np.random.uniform(0.1, 1.0)  # Simulación de sudoración

            # Guardar los datos en Firebase
            data = {
                'emotion': emotion,
                'confidence': confidence,
                'bpm': bpm,
                'sweating': sweat_level,
                'date': datetime.now().isoformat()
            }
            db.collection('emotion_data').add(data)

            # Devolver la respuesta con la emoción detectada
            return jsonify({'emotion': emotion, 'confidence': confidence, 'bpm': bpm, 'sweat_level': sweat_level}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)