from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

# Cargar el modelo entrenado
model = load_model('sports_classifier.h5')

# Definir manualmente las clases en el orden correcto
class_labels = {
    0: 'futbol',
    1: 'baloncesto',
    2: 'natacion',
    3: 'ajedrez',
    4: 'hockey',
    5: 'formula1',
    6: 'esgrima',
    7: 'boxeo',
    8: 'tiro',
    9: 'tenis'
}

# Cargar y preprocesar la imagen de prueba
test_image = load_img('single_test/ajedrez.jpg', target_size=(128, 128))
test_image = img_to_array(test_image)
test_image = test_image / 255.0
test_image = np.expand_dims(test_image, axis=0)

# Predecir la clase
result = model.predict(test_image)

# Mostrar las probabilidades para depurar
print(f"Probabilidades predichas: {result}")

# Obtener el índice de la clase con mayor probabilidad
predicted_class_index = np.argmax(result)

# Obtener el nombre de la clase predicha
predicted_class_label = class_labels[predicted_class_index]

# Mostrar resultado
print(f'La imagen pertenece a la clase: {predicted_class_label}')
