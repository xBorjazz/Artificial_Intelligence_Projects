import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import numpy as np

# ================================
# Definición del grafo y parámetros
# ================================
graph = {
    'A': {'B': 2, 'C': 4},
    'B': {'A': 2, 'C': 1, 'D': 7},
    'C': {'A': 4, 'B': 1, 'E': 3},
    'D': {'B': 7, 'E': 2, 'F': 5},
    'E': {'C': 3, 'D': 2, 'F': 6},
    'F': {'D': 5, 'E': 6}
}

# Creamos el grafo de NetworkX y definimos posiciones para los nodos
G = nx.Graph()
for node in graph:
    for neighbor, weight in graph[node].items():
        G.add_edge(node, neighbor, weight=weight)
pos = nx.spring_layout(G)

# Parámetros del ACO
alpha = 1.0      # Influencia de la feromona
beta = 2.0       # Influencia del valor heurístico (1/distancia)
rho = 0.5        # Tasa de evaporación
Q = 100          # Constante para depósito de feromonas
num_ants = 10    # Número de hormigas por iteración
num_iterations = 20  # Número de iteraciones

# Inicialización de feromonas: todas las aristas parten de 0.1
pheromones = {i: {j: 0.1 for j in graph[i]} for i in graph}

# Valor heurístico: inversa de la distancia
heuristic = {i: {j: 1.0/graph[i][j] for j in graph[i]} for i in graph}

# ================================
# Funciones del algoritmo ACO
# ================================
def choose_next_node(current, visited):
    """Elige el siguiente nodo basado en feromonas y heurística."""
    neighbors = list(graph[current].keys())
    allowed = [node for node in neighbors if node not in visited]
    if not allowed:
        return None
    probs = []
    for j in allowed:
        tau = pheromones[current][j] ** alpha
        eta = heuristic[current][j] ** beta
        probs.append(tau * eta)
    total = sum(probs)
    if total == 0:
        return random.choice(allowed)
    probs = [p/total for p in probs]
    next_node = random.choices(allowed, weights=probs, k=1)[0]
    return next_node

def construct_solution():
    """Construye un camino desde A hasta F para una hormiga."""
    start = 'A'
    end = 'F'
    current = start
    path = [current]
    while current != end:
        next_node = choose_next_node(current, path)
        if next_node is None:
            return None
        path.append(next_node)
        current = next_node
    return path

def path_cost(path):
    """Calcula el costo total de un camino."""
    cost = 0
    for i in range(len(path)-1):
        cost += graph[path[i]][path[i+1]]
    return cost

# ================================
# Simulación del ACO
# ================================
# Almacenaremos los resultados de cada iteración: caminos de cada hormiga, el mejor camino y su costo.
simulation_results = []
best_path_global = None
best_cost_global = float('inf')

for it in range(num_iterations):
    ant_paths = []
    for ant in range(num_ants):
        sol = construct_solution()
        if sol is not None:
            ant_paths.append(sol)
    # Evaporación de feromonas
    for i in pheromones:
        for j in pheromones[i]:
            pheromones[i][j] *= (1 - rho)
    # Depósito de feromonas según la calidad del camino
    for path in ant_paths:
        cost = path_cost(path)
        delta = Q / cost
        for i in range(len(path)-1):
            a = path[i]
            b = path[i+1]
            pheromones[a][b] += delta
            pheromones[b][a] += delta
        if cost < best_cost_global:
            best_cost_global = cost
            best_path_global = path
    simulation_results.append({
        'iteration': it + 1,
        'ant_paths': ant_paths,
        'best_path': best_path_global,
        'best_cost': best_cost_global
    })

# ================================
# Preparación de datos para la animación
# ================================
def get_positions_for_path(path, pos, steps_per_edge=20):
    """
    Dado un camino (lista de nodos), devuelve una lista de posiciones (x,y) interpoladas
    para simular el movimiento en cada arista.
    """
    positions = []
    for i in range(len(path)-1):
        start = np.array(pos[path[i]])
        end = np.array(pos[path[i+1]])
        # Se generan 'steps_per_edge' posiciones intermedias para el segmento
        for t in np.linspace(0, 1, steps_per_edge, endpoint=False):
            positions.append((1-t)*start + t*end)
    positions.append(np.array(pos[path[-1]]))
    return positions

# Para cada iteración, calculamos las trayectorias de cada hormiga
animation_data = []
for res in simulation_results:
    ant_trajectories = []
    for path in res['ant_paths']:
        traj = get_positions_for_path(path, pos, steps_per_edge=20)
        ant_trajectories.append(traj)
    animation_data.append({
        'iteration': res['iteration'],
        'ant_trajectories': ant_trajectories,
        'best_path': res['best_path'],
        'best_cost': res['best_cost'],
        'max_frames': max([len(traj) for traj in ant_trajectories]) if ant_trajectories else 0
    })

# Se mapea cada cuadro global a (iteración, cuadro local)
frames_per_iteration = [data['max_frames'] for data in animation_data]
total_frames = sum(frames_per_iteration)
frame_mapping = []
for it, frames in enumerate(frames_per_iteration):
    for local in range(frames):
        frame_mapping.append((it, local))

# ================================
# Configuración de la animación con Matplotlib
# ================================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title("ACO - Simulación del recorrido de las hormigas")
# Dibujar el grafo base
nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray')
nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=500)
nx.draw_networkx_labels(G, pos, ax=ax, font_size=12)

# Elementos gráficos para actualizar:
# - Puntos que representan a las hormigas
# - Línea que resalta el mejor camino
# - Texto con información de la iteración
ant_scatter = ax.scatter([], [], s=100, color='red', zorder=3)
best_path_line, = ax.plot([], [], color='green', linewidth=3, zorder=2, label='Mejor camino')
iteration_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=14, verticalalignment='top')

def update(frame):
    it, local = frame_mapping[frame]
    data = animation_data[it]
    iteration_text.set_text(f"Iteración {data['iteration']} - Mejor camino: {data['best_path']} (Costo: {data['best_cost']})")
    
    # Para cada hormiga, se actualiza su posición según el cuadro local
    positions = []
    for traj in data['ant_trajectories']:
        if local < len(traj):
            pos_ant = traj[local]
        else:
            pos_ant = traj[-1]
        positions.append(pos_ant)
    
    if positions:
        pos_array = np.vstack(positions)
        ant_scatter.set_offsets(pos_array)
    else:
        ant_scatter.set_offsets([])
    
    # Se resalta el mejor camino encontrado en la iteración (se dibujan las aristas del camino)
    if data['best_path']:
        path_nodes = data['best_path']
        x_coords = [pos[node][0] for node in path_nodes]
        y_coords = [pos[node][1] for node in path_nodes]
        best_path_line.set_data(x_coords, y_coords)
    else:
        best_path_line.set_data([], [])
    
    return ant_scatter, best_path_line, iteration_text

ani = FuncAnimation(fig, update, frames=total_frames, interval=100, blit=False, repeat=False)
plt.legend()
plt.show()
