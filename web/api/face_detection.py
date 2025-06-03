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
from flask import Flask, request, jsonify
from datetime import datetime
import numpy as np
import cv2
import os
from deepface import DeepFace
# from firebase_admin import firestore, initialize_app

app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Verificar campos de formulario
        bpm = request.form.get('bpm')
        gsr = request.form.get('gsr')
        file = request.files.get('image')

        if not bpm or not gsr:
            return jsonify({'error': 'Faltan bpm o gsr'}), 400
        if not file:
            return jsonify({'error': 'No se recibió imagen'}), 400

        # Leer la imagen como arreglo numpy
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Guardar la imagen localmente
        os.makedirs('./img', exist_ok=True)
        img_path = f'./img/{datetime.now().strftime("%Y%m%d%H%M%S%f")}.jpg'
        cv2.imwrite(img_path, img)

        # Convertir a escala de grises y detectar rostros
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return jsonify({'error': 'No se detectaron rostros en la imagen'}), 400

        # Analizar el primer rostro detectado
        (x, y, w, h) = faces[0]
        face_roi = img[y:y + h, x:x + w]

        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]

        emotion = result['dominant_emotion']
        confidence = result['emotion'][emotion]

        # Preparar datos
        data = {
            'emotion': emotion,
            'confidence': confidence,
            'bpm': float(bpm),
            'sweating': float(gsr),
            'date': datetime.now().isoformat()
        }

        db.collection('emotion_data').add(data)  # Descomenta si usas Firebase

        return jsonify(data), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
