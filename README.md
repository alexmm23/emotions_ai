# Sistema de Detección de Emociones en Tiempo Real

Este proyecto implementa un sistema completo de detección de emociones faciales usando inteligencia artificial, que combina análisis de expresiones faciales con datos de sensores biométricos (BPM y GSR).

## 🚀 Características

- **Detección de emociones faciales** usando DeepFace
- **API REST** para procesamiento de imágenes
- **Integración con Firebase** para almacenamiento de datos
- **Análisis en tiempo real** con cámara web
- **Modelo personalizado** entrenado con dataset FER2013
- **Métricas biométricas** (BPM y conductividad de la piel)

## 📋 Requisitos del Sistema

### Software
- Python 3.8 o superior
- OpenCV 4.x
- TensorFlow 2.x
- Flask
- Firebase Admin SDK

### Hardware
- Cámara web
- Sensores biométricos (opcional)
- GPU recomendada para entrenamiento

## 🛠️ Instalación

1. **Clonar el repositorio**
```bash
git clone <url-del-repositorio>
cd IoT
```

2. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

3. **Configurar Firebase**
   - Crear proyecto en [Firebase Console](https://console.firebase.google.com/)
   - Descargar `firebase_credentials.json`
   - Colocar el archivo en la carpeta raíz del proyecto

4. **Estructura de directorios**
```
IoT/
├── web/
│   └── api/
│       └── face_detection.py
├── modelo_emociones.py
├── firebase_credentials.json
├── requirements.txt
└── README.md
```

## 📊 Dataset

El proyecto utiliza el dataset FER2013 organizado en carpetas:

```
train/
├── anger/
├── disgust/
├── fear/
├── happiness/
├── sadness/
├── surprise/
└── neutral/
```

## 🔧 Uso

### 1. Entrenar el Modelo
```bash
python modelo_emociones.py
```

### 2. Ejecutar la API
```bash
cd web/api
python face_detection.py
```

### 3. Probar con Postman

**Endpoint:** `POST http://localhost:5000/upload`

**Parámetros (form-data):**
- `image`: Archivo de imagen (JPG/PNG)
- `bpm`: Frecuencia cardíaca (número)
- `gsr`: Conductividad de la piel (número)

**Respuesta de ejemplo:**
```json
{
    "emotion": "happy",
    "confidence": 95.2,
    "bpm": 75.0,
    "sweating": 0.45,
    "date": "2025-06-17T14:30:00.000Z"
}
```

## 🧠 Arquitectura del Modelo

### Red Neuronal Convolucional (CNN)
- **Entrada:** Imágenes 48x48 píxeles en escala de grises
- **Capas convolucionales:** 3 bloques (32, 64, 128 filtros)
- **Regularización:** Dropout (0.25 y 0.5)
- **Salida:** 7 emociones con probabilidades (softmax)

### Emociones Detectadas
1. Anger (Enojo)
2. Disgust (Asco)
3. Fear (Miedo)
4. Happiness (Felicidad)
5. Sadness (Tristeza)
6. Surprise (Sorpresa)
7. Neutral (Neutral)

## 📈 Monitoreo y Almacenamiento

Los datos se almacenan en Firebase Firestore con la siguiente estructura:

```json
{
    "emotion": "string",
    "confidence": "number",
    "bpm": "number",
    "sweating": "number",
    "date": "ISO string"
}
```

## 🔍 Estructura de Archivos

| Archivo | Descripción |
|---------|-------------|
| `modelo_emociones.py` | Script de entrenamiento del modelo |
| `web/api/face_detection.py` | API Flask para detección |
| `firebase_credentials.json` | Credenciales de Firebase |
| `modelo_facial.h5` | Modelo entrenado (generado) |

## 🚨 Solución de Problemas

### Error: "No se detectaron rostros"
- Verificar iluminación adecuada
- Usar imágenes claras y centradas
- Ajustar parámetros de `detectMultiScale`

### Error: "Firebase credentials"
- Verificar ruta del archivo `firebase_credentials.json`
- Confirmar permisos de Firestore

### Error: "Model not found"
- Ejecutar primero `modelo_emociones.py` para generar el modelo

## 🤝 Contribuciones

1. Fork el proyecto
2. Crear rama para nueva funcionalidad
3. Commit de cambios
4. Push a la rama
5. Crear Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.

## 👥 Autores

- Tu Nombre - Desarrollo principal
- Contacto: tu.email@example.com

## 🙏 Agradecimientos

- Dataset FER2013
- Biblioteca DeepFace
- TensorFlow/Keras
- OpenCV Community