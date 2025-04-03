import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

# Cargar el dataset desde carpetas
def load_images_from_folders(dataset_path):
    emotion_labels = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    faces = []
    emotions = []
    
    print("Cargando imágenes desde carpetas...")
    
    for idx, emotion in enumerate(emotion_labels):
        emotion_dir = os.path.join(dataset_path, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Advertencia: No se encontró carpeta para {emotion}. Saltando.")
            continue
            
        print(f"Procesando imágenes de: {emotion}")
        image_files = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in tqdm(image_files):
            img_path = os.path.join(emotion_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                if img.shape != (48, 48):
                    img = cv2.resize(img, (48, 48))
                
                img = np.expand_dims(img, axis=-1)
                img = img / 255.0
                
                faces.append(img)
                
                emotion_one_hot = np.zeros(len(emotion_labels))
                emotion_one_hot[idx] = 1
                emotions.append(emotion_one_hot)
            else:
                print(f"No se pudo cargar: {img_path}")
    
    return np.array(faces), np.array(emotions)

# Cargar los datos
X_faces, y_emotions = load_images_from_folders('train')  
print(f"Datos cargados: {X_faces.shape[0]} imágenes en total")

# Dividir en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(
    X_faces, y_emotions, test_size=0.2, random_state=42
)

# Definir modelo que usa solo imágenes faciales
input_face = layers.Input(shape=(48, 48, 1))
x = layers.Conv2D(32, (3, 3), activation='relu')(input_face)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.25)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation='relu')(x)
output = layers.Dense(7, activation='softmax')(x)

model = models.Model(inputs=input_face, outputs=output)
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# Callbacks para mejorar el entrenamiento
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

# Entrenar el modelo
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=callbacks
)

# Guardar el modelo
model.save('modelo_facial.h5')
print("Modelo guardado como 'modelo_facial.h5'")

# Visualizar resultados
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Precisión del modelo')
plt.ylabel('Precisión')
plt.xlabel('Época')
plt.legend(['Entrenamiento', 'Validación'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Pérdida del modelo')
plt.ylabel('Pérdida')
plt.xlabel('Época')
plt.legend(['Entrenamiento', 'Validación'], loc='upper right')

plt.tight_layout()
plt.savefig('training_results_facial.png')
plt.show()