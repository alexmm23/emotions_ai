# Documentación del Entrenamiento del Modelo de Emociones

## 📖 Descripción General

Este documento detalla el proceso de entrenamiento del modelo de reconocimiento de emociones faciales utilizando redes neuronales convolucionales (CNN) con el dataset FER2013.

## 🎯 Objetivo

Entrenar un modelo de deep learning capaz de clasificar expresiones faciales en 7 categorías emocionales diferentes con alta precisión y robustez.

## 📊 Dataset

### FER2013 (Facial Expression Recognition 2013)
- **Origen:** Kaggle Competition
- **Tamaño:** ~35,000 imágenes
- **Resolución:** 48x48 píxeles
- **Formato:** Escala de grises
- **Clases:** 7 emociones

### Distribución de Clases
| Emoción | Cantidad Aproximada | Porcentaje |
|---------|-------------------|------------|
| Happiness | ~8,989 | 25.7% |
| Neutral | ~6,198 | 17.7% |
| Sadness | ~6,077 | 17.4% |
| Anger | ~4,953 | 14.2% |
| Surprise | ~4,002 | 11.4% |
| Fear | ~5,121 | 14.6% |
| Disgust | ~547 | 1.6% |

## 🏗️ Arquitectura del Modelo

### Diseño de la Red Neuronal

```python
# Arquitectura CNN
input_face = Input(shape=(48, 48, 1))

# Bloque Convolucional 1
x = Conv2D(32, (3, 3), activation='relu')(input_face)
x = MaxPooling2D((2, 2))(x)

# Bloque Convolucional 2
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

# Bloque Convolucional 3
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

# Regularización
x = Dropout(0.25)(x)
x = Flatten()(x)

# Capas Densas
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)

# Capa de Salida
output = Dense(7, activation='softmax')(x)
```

### Parámetros del Modelo
- **Parámetros totales:** ~2.3M
- **Parámetros entrenables:** ~2.3M
- **Memoria requerida:** ~120MB

## ⚙️ Configuración de Entrenamiento

### Hiperparámetros
```python
# Compilación del modelo
optimizer = 'adam'
loss = 'categorical_crossentropy'
metrics = ['accuracy']
learning_rate = 0.001  # (por defecto de Adam)

# Entrenamiento
epochs = 50
batch_size = 64
validation_split = 0.2
```

### Callbacks Implementados

#### 1. Early Stopping
```python
EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```
- **Propósito:** Evitar sobreentrenamiento
- **Monitoreo:** Pérdida de validación
- **Paciencia:** 10 épocas sin mejora

#### 2. Reduce Learning Rate on Plateau
```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)
```
- **Propósito:** Optimización adaptativa
- **Reducción:** 50% cuando se estanca
- **LR mínimo:** 1e-6

## 📈 Proceso de Entrenamiento

### 1. Preprocesamiento de Datos

```python
def preprocess_image(image):
    # Redimensionar a 48x48
    image = cv2.resize(image, (48, 48))
    
    # Normalizar valores de píxeles [0, 1]
    image = image / 255.0
    
    # Añadir dimensión de canal
    image = np.expand_dims(image, axis=-1)
    
    return image
```

### 2. Aumento de Datos (Data Augmentation)
Aunque no implementado en la versión actual, se recomienda:
- Rotación: ±15°
- Zoom: ±10%
- Desplazamiento horizontal: ±10%
- Volteo horizontal: Falso (las emociones son asimétricas)

### 3. División de Datos
- **Entrenamiento:** 80% (28,000 imágenes aprox.)
- **Validación:** 20% (7,000 imágenes aprox.)
- **Prueba:** Validación cruzada en conjunto de validación

## 📊 Métricas de Evaluación

### Métricas Principales
1. **Accuracy (Precisión Global)**
   - Fórmula: (TP + TN) / (TP + TN + FP + FN)
   - Objetivo: >85%

2. **Loss (Pérdida Categórica)**
   - Función: Categorical Crossentropy
   - Objetivo: <0.5

3. **Precision por Clase**
   - Importante para clases desbalanceadas (especialmente 'disgust')

4. **Recall por Clase**
   - Crítico para aplicaciones médicas/psicológicas

### Matriz de Confusión
Se genera automáticamente para analizar:
- Confusiones entre emociones similares
- Rendimiento por clase individual
- Identificación de sesgos del modelo

## 🎯 Resultados Esperados

### Rendimiento Objetivo
- **Precisión de Entrenamiento:** 90-95%
- **Precisión de Validación:** 85-90%
- **Tiempo de Entrenamiento:** 2-4 horas (CPU) / 30-60 min (GPU)

### Curvas de Aprendizaje
El script genera automáticamente:
- Gráfico de precisión vs. épocas
- Gráfico de pérdida vs. épocas
- Comparación entrenamiento vs. validación

## 🔧 Optimizaciones Implementadas

### 1. Dropout Regularization
- **Capa 1:** 25% después de convoluciones
- **Capa 2:** 50% en capas densas
- **Propósito:** Reducir overfitting

### 2. Batch Normalization (Recomendado)
```python
# Para futuras mejoras
x = BatchNormalization()(x)
```

### 3. Transfer Learning (Opcional)
- Usar modelos preentrenados como base
- VGG16, ResNet50 adaptados para emociones

## 🚀 Ejecución del Entrenamiento

### Comando de Ejecución
```bash
python modelo_emociones.py
```

### Requisitos de Hardware
- **RAM mínima:** 8GB
- **Espacio en disco:** 5GB
- **GPU (opcional):** GTX 1060 o superior
- **Tiempo estimado:** 2-4 horas

### Archivos Generados
- `modelo_facial.h5` - Modelo entrenado
- `training_results_facial.png` - Gráficos de entrenamiento
- Logs de entrenamiento en consola

## 📋 Checklist de Entrenamiento

- [ ] Dataset descargado y organizado
- [ ] Dependencias instaladas
- [ ] Estructura de carpetas correcta
- [ ] Suficiente espacio en disco
- [ ] Ejecutar script de entrenamiento
- [ ] Validar métricas obtenidas
- [ ] Guardar modelo final
- [ ] Documentar resultados

## 🔍 Troubleshooting

### Problemas Comunes

#### 1. Memory Error
```bash
# Reducir batch_size
batch_size = 32  # en lugar de 64
```

#### 2. Slow Training
```bash
# Verificar uso de GPU
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

#### 3. Poor Convergence
- Ajustar learning rate
- Modificar arquitectura
- Aumentar datos de entrenamiento

### Logs de Debug
```python
# Activar logging detallado
import logging
logging.basicConfig(level=logging.INFO)
```

## 📚 Referencias

1. [FER2013 Paper](https://arxiv.org/abs/1307.0414)
2. [CNN for Emotion Recognition](https://arxiv.org/abs/1710.07557)
3. [TensorFlow Documentation](https://www.tensorflow.org/)
4. [Keras Functional API](https://keras.io/guides/functional_api/)

## 📞 Soporte

Para problemas con el entrenamiento:
1. Revisar logs de error
2. Verificar versiones de librerías
3. Consultar documentación de TensorFlow
4. Contactar al equipo de desarrollo

---

**Última actualización:** Junio 2025
**Versión del modelo:** 1.0
**Autor:** [Tu Nombre]