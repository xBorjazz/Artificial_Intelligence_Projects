import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.33333, epochs=8000):
        self.W = np.zeros(input_size + 1)  # +1 for the bias term
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)  # Insert bias term
        z = self.W.T.dot(x)
        return self.activation_function(z)

    def train(self, X, y):
        for _ in range(self.epochs):
            for inputs, label in zip(X, y):
                prediction = self.predict(inputs)
                self.W += self.learning_rate * (label - prediction) * np.insert(inputs, 0, 1)

    def print_weights(self):
        print(f"Pesos después del entrenamiento: {self.W}")

# Datos de entrada y salidas deseadas para la compuerta OR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 1])

# Crear el perceptrón y entrenarlo
perceptron = Perceptron(input_size=2)
perceptron.train(X, y)

# Imprimir los pesos después del entrenamiento
perceptron.print_weights()

# Probar el perceptrón
for inputs in X:
    print(f"Entrada: {inputs} -> Salida predicha: {perceptron.predict(inputs)}")
