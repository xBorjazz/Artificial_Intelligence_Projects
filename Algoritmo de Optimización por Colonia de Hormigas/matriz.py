import numpy as np

INF = float('inf')  # Representa infinito

adj_matrix = np.array([
    [0, 2, 4, INF, INF, INF],  # A
    [2, 0, 1, 7, INF, INF],    # B
    [4, 1, 0, INF, 3, INF],    # C
    [INF, 7, INF, 0, 2, 5],    # D
    [INF, INF, 3, 2, 0, 6],    # E
    [INF, INF, INF, 5, 6, 0]   # F
])

print(adj_matrix)
