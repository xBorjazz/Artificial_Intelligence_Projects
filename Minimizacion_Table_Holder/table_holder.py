import numpy as np
import matplotlib.pyplot as plt
import random

# Función Holder Table
def holder_table(x, y):
    return -abs(np.sin(x) * np.cos(y) * np.exp(abs(1 - np.sqrt(x**2 + y**2)/np.pi)))

# Parámetros del algoritmo ABC
NUM_BEES = 50
NUM_ITER = 100
LIMIT = 20
BOUND = [-10, 10]

# Generar soluciones iniciales
def init_population():
    return [(random.uniform(*BOUND), random.uniform(*BOUND)) for _ in range(NUM_BEES)]

# Evaluar población
def evaluate(pop):
    return [holder_table(x, y) for x, y in pop]

# Movimiento de abejas obreras
def neighbor(sol):
    phi = random.uniform(-1, 1)
    index = random.randint(0, 1)
    new_sol = list(sol)
    new_sol[index] += phi * (sol[index] - random.uniform(*BOUND))
    new_sol[index] = np.clip(new_sol[index], *BOUND)
    return tuple(new_sol)

# Algoritmo ABC
def abc():
    population = init_population()
    fitness = evaluate(population)
    trial = [0] * NUM_BEES

    best_index = np.argmin(fitness)
    best_sol = population[best_index]
    best_val = fitness[best_index]

    for _ in range(NUM_ITER):
        for i in range(NUM_BEES):
            new_sol = neighbor(population[i])
            new_fit = holder_table(*new_sol)
            if new_fit < fitness[i]:
                population[i] = new_sol
                fitness[i] = new_fit
                trial[i] = 0
            else:
                trial[i] += 1

        # Abejas exploradoras
        for i in range(NUM_BEES):
            if trial[i] > LIMIT:
                population[i] = (random.uniform(*BOUND), random.uniform(*BOUND))
                fitness[i] = holder_table(*population[i])
                trial[i] = 0

        curr_best_index = np.argmin(fitness)
        if fitness[curr_best_index] < best_val:
            best_val = fitness[curr_best_index]
            best_sol = population[curr_best_index]

    return best_sol, best_val

# Ejecutar el algoritmo ABC
best_solution, best_value = abc()
print("Mejor solución encontrada:", best_solution)
print("Valor mínimo de la función:", best_value)

# Graficar la función y la solución
x = np.linspace(BOUND[0], BOUND[1], 400)
y = np.linspace(BOUND[0], BOUND[1], 400)
X, Y = np.meshgrid(x, y)
Z = holder_table(X, Y)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.scatter(best_solution[0], best_solution[1], best_value, color='r', s=50, label='Mínimo encontrado')
ax.set_title('Función Holder Table con mínimo encontrado')
ax.legend()
plt.show()
