import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np

# Parámetros
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 500  # Número máximo de épocas (EarlyStopping lo detendrá antes si deja de mejorar)
NUM_CLASSES = 10  # Asegúrate de que coincide con el número de carpetas en "dataset"

# Preprocesamiento y separación automática de entrenamiento y validación
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,  # Mayor rango de rotación
    width_shift_range=0.3,  # Mayor rango de desplazamiento horizontal
    height_shift_range=0.3,  # Mayor rango de desplazamiento vertical
    shear_range=0.3,  # Mayor deformación
    zoom_range=0.3,  # Mayor zoom
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 80% entrenamiento, 20% validación
)

# Generador para el conjunto de entrenamiento
train_generator = datagen.flow_from_directory(
    'dataset',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Generador para el conjunto de validación
validation_generator = datagen.flow_from_directory(
    'dataset',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Imprimir el mapeo de clases para conocer el orden (usado luego en predicción)
print(f"Clases detectadas: {train_generator.class_indices}")

# Definir la arquitectura del modelo CNN
model = Sequential()

# Primera capa convolucional
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Segunda capa convolucional
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Tercera capa convolucional
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Cuarta capa convolucional (para aumentar la profundidad)
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Aplanar y añadir capas densas
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # Dropout para reducir el sobreajuste
model.add(Dense(NUM_CLASSES, activation='softmax'))

# Configurar el optimizador (Adam con tasa de aprendizaje inicial más baja)
optimizer = Adam(learning_rate=0.001)

# Compilar el modelo
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# EarlyStopping para detener el entrenamiento si deja de mejorar
early_stopping = EarlyStopping(
    monitor='val_loss',  # Basado en la pérdida de validación
    patience=10,  # Si no mejora después de 10 épocas, se detiene
    restore_best_weights=True  # Restaura los mejores pesos
)

# ReduceLROnPlateau para reducir la tasa de aprendizaje si el modelo se estanca
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # Reduce la tasa de aprendizaje a la mitad
    patience=5,  # Si no mejora después de 5 épocas, reduce la tasa
    min_lr=0.00001  # Tasa de aprendizaje mínima
)

# Entrenar el modelo con EarlyStopping y ReduceLROnPlateau
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Guardar el modelo entrenado
model.save('sports_classifier.h5')

# Mostrar resultados finales de precisión y pérdida
import matplotlib.pyplot as plt

# Precisión
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()

# Pérdida
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()
