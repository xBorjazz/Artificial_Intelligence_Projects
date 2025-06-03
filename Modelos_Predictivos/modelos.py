import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
import matplotlib.pyplot as plt

# Cargar datos
empleados = pd.read_csv('salarios.csv')
prediccion = pd.read_excel('prediccion.xlsx')

# Separar características y etiquetas
X = empleados.drop('Salary', axis=1)
y = empleados['Salary']

# Columnas categóricas y numéricas
cat_cols = ['Gender', 'Education Level', 'Job Title']
num_cols = ['Age', 'Years of Experience']

# Pipeline de preprocesamiento
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(sparse_output=False), cat_cols)
])

# Transformar datos
X_preprocessed = preprocessor.fit_transform(X)
X_pred_preprocessed = preprocessor.transform(prediccion[X.columns])

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Construcción del modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenamiento
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=16, verbose=0)

# Resultados
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.2f}")

# Gráficas de pérdida
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Curva de Pérdida")
plt.xlabel("Épocas")
plt.ylabel("MSE")
plt.legend()
plt.show()

# Predicción para nuevos empleados
y_pred = model.predict(X_pred_preprocessed).flatten()
prediccion['Predicted Salary'] = y_pred

# Calcular error porcentual
prediccion['Error %'] = 100 * abs(prediccion['Predicted Salary'] - prediccion['Desired salary']) / prediccion['Desired salary']
print(prediccion[['Predicted Salary', 'Desired salary', 'Error %']])

# Guardar predicciones
prediccion.to_csv("salarios_predichos.csv", index=False)
