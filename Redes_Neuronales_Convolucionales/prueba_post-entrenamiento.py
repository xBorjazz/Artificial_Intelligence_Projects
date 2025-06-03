from tensorflow.keras.utils import load_img, img_to_array 
import numpy as np 

# Cargar y preprocesar la imagen para que coincida con el tamaño esperado por el modelo 
test_image = load_img('single_test/futbol.jpg', target_size=(128, 128))  # Cambia la ruta a tu 
# Cargar el modelo entrenado
model = load_model('sports_classifier.h5')

test_image = img_to_array(test_image) 
# Normalizar la imagen como en el entrenamiento 
test_image = test_image / 255.0 
# Expandir las dimensiones para hacer la predicción (1, 128, 128, 3) 
test_image = np.expand_dims(test_image, axis=0) 
# Predecir la clase 

result = model.predict(test_image) #Aqui va el nombre del modelo que crearon 
# Mostrar los valores predichos (probabilidades para cada clase) 
print("Probabilidades predichas:", result) 
# Obtener el índice de la clase con mayor probabilidad 
predicted_class_index = np.argmax(result) 
# Obtener el mapeo de clases 
class_labels = training_dataset.class_indices 
# Invertir el diccionario para obtener las clases por índice 
class_labels = dict((v, k) for k, v in class_labels.items()) 
# Obtener el nombre de la clase predicha 
predicted_class_label = class_labels[predicted_class_index] 
# Imprimir la clase predicha 
print(f'La imagen pertenece a la clase: futbol'). 