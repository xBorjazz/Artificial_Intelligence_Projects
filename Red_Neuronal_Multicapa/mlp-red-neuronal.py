import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Asegurar que matplotlib se renderiza en VS Code
matplotlib.use('TkAgg')  # Cambia esto si necesitas otro backend compatible con tu sistema

# Cargar los datos
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Verificar la cantidad de columnas en test.csv
print("Shape de test.csv:", test_df.shape)  # Debe ser (num_muestras, 784)

# Separar imágenes y etiquetas para el conjunto de entrenamiento
X_train = train_df.iloc[:, 1:].values.astype("float32") / 255.0  # Normalizar datos
y_train = to_categorical(train_df.iloc[:, 0].values, num_classes=10)

# Cargar todas las 784 columnas de test.csv (sin etiquetas)
X_test = test_df.iloc[:, :].values.astype("float32") / 255.0  # Asegurar que se usan todas las columnas
print("Shape de X_test después de cargar:", X_test.shape)  # Debe ser (num_muestras, 784)

# Separar datos de validación
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Definir modelo MLP
def create_model(optimizer='adam'):
    model = Sequential([
        Flatten(input_shape=(784,)),         # Capa de entrada con 784 características
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')         # Capa de salida para clasificación en 10 clases
    ])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Entrenar y evaluar con diferentes optimizadores
optimizers = ['SGD', 'Adam', 'RMSprop', 'Nadam', 'Adadelta']
history_dict = {}

for opt in optimizers:
    print(f"\nEntrenando con optimizador: {opt}")
    model = create_model(optimizer=opt)
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=1)
    history_dict[opt] = history

# Evaluar en test (no se puede calcular precisión porque no hay etiquetas en test.csv)
results = {}
for opt in optimizers:
    model = create_model(optimizer=opt)
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    loss = model.evaluate(X_test, verbose=0)  # Se evalúa solo la pérdida
    results[opt] = loss
    print(f"Pérdida en test con {opt}: {loss:.4f}")

# Graficar resultados de entrenamiento (precisión de validación)
plt.figure(figsize=(10, 5))
for opt, history in history_dict.items():
    plt.plot(history.history['val_accuracy'], label=f"{opt}")
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.title("Comparación de optimizadores")
plt.legend()
plt.show(block=True)  # Evita que la ventana se cierre inmediatamente

# Mostrar imágenes de prueba con predicciones
num_images = 5
random_indices = np.random.choice(len(X_test), num_images, replace=False)

for i in random_indices:
    plt.figure(figsize=(4, 4))  # Ajustar tamaño de la figura
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")  # Escala de grises
    pred = model.predict(X_test[i].reshape(1, -1))
    plt.title(f"Predicción: {np.argmax(pred)}", fontsize=14)
    plt.axis("off")
    plt.show(block=True)  # Muestra la imagen sin cerrar la ventana
    plt.pause(0.5)  # Agrega una pausa para ver cada imagen antes de la siguiente
