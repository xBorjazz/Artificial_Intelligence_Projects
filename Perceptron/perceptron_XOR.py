import numpy as np

# Función de activación sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función sigmoide (para backpropagation)
def sigmoid_derivative(x):
    return x * (1 - x)

# Datos de entrada (XOR)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Salidas deseadas (XOR)
y = np.array([[0], [1], [1], [0]])

# Inicialización de pesos y configuración
np.random.seed(42)  # Para resultados reproducibles
input_layer_size = 2
hidden_layer_size = 4
output_layer_size = 1

# Pesos para las conexiones
weights_input_hidden = np.random.uniform(-1, 1, (input_layer_size, hidden_layer_size))
weights_hidden_output = np.random.uniform(-1, 1, (hidden_layer_size, output_layer_size))

# Bias para cada capa
bias_hidden = np.random.uniform(-1, 1, (1, hidden_layer_size))
bias_output = np.random.uniform(-1, 1, (1, output_layer_size))

# Tasa de aprendizaje
learning_rate = 0.0001

# Entrenamiento
epochs = 42
for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Cálculo del error
    error = y - predicted_output

    # Backpropagation
    output_layer_gradient = error * sigmoid_derivative(predicted_output)
    hidden_layer_error = np.dot(output_layer_gradient, weights_hidden_output.T)
    hidden_layer_gradient = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

    # Actualización de pesos y bias
    weights_hidden_output += np.dot(hidden_layer_output.T, output_layer_gradient) * learning_rate
    bias_output += np.sum(output_layer_gradient, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += np.dot(X.T, hidden_layer_gradient) * learning_rate
    bias_hidden += np.sum(hidden_layer_gradient, axis=0, keepdims=True) * learning_rate

    # Opción: imprimir el error cada cierto número de epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")

# Resultados después del entrenamiento
print("\nResultados después del entrenamiento:")
for inputs, label in zip(X, y):
    # Forward pass para una predicción final
    hidden_layer_output = sigmoid(np.dot(inputs, weights_input_hidden) + bias_hidden)
    output = sigmoid(np.dot(hidden_layer_output, weights_hidden_output) + bias_output)
    
    # Convertir el valor predicho a un escalar
    predicted_value = output[0, 0]
    print(f"Entrada: {inputs} -> Salida predicha: {predicted_value:.3f} (Esperada: {label[0]})")
