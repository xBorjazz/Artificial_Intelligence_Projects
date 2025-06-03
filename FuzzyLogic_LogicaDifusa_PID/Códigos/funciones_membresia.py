import numpy as np
import matplotlib.pyplot as plt

def triangular(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

def trapezoidal(x, a, b, c, d):
    return np.clip(np.minimum((x - a) / (b - a), (d - x) / (d - c)), 0, 1)

# Rango común
x = np.linspace(-3, 3, 500)

# Funciones para el controlador PI
membership_PI = {
    "Error - NB": triangular(x, -3, -3, -2),
    "Error - NM": triangular(x, -3, -2, -1),
    "Error - NS": triangular(x, -2, -1, 0),
    "Error - ZO": triangular(x, -1, 0, 1),
    "Error - PS": triangular(x, 0, 1, 2),
    "Error - PM": triangular(x, 1, 2, 3),
    "Error - PB": triangular(x, 2, 3, 3),
    "CE - NS": triangular(x, -2, -1, 0),
    "CE - ZO": triangular(x, -1, 0, 1),
    "CE - PS": triangular(x, 0, 1, 2),
    "Salida - NB": triangular(x, -3, -2, -1),
    "Salida - ZO": triangular(x, -1, 0, 1),
    "Salida - PB": triangular(x, 1, 2, 3),
}

# Funciones para el controlador D
membership_D = {
    "Derivada - NB": triangular(x, -3, -3, -2),
    "Derivada - NS": triangular(x, -3, -2, 0),
    "Derivada - ZO": triangular(x, -1, 0, 1),
    "Derivada - PS": triangular(x, 0, 2, 3),
    "Derivada - PB": triangular(x, 2, 3, 3),
    "Salida D - NB": triangular(x, -3, -2, -1),
    "Salida D - ZO": triangular(x, -1, 0, 1),
    "Salida D - PB": triangular(x, 1, 2, 3),
}

def plot_all_memberships(membership_dict, title):
    n = len(membership_dict)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(14, rows * 3))
    fig.suptitle(title, fontsize=16)

    axs = axs.flatten()
    for idx, (name, y_values) in enumerate(membership_dict.items()):
        axs[idx].plot(x, y_values, linewidth=2)
        axs[idx].set_title(name)
        axs[idx].set_ylim(0, 1.1)
        axs[idx].grid(True, linestyle='--', alpha=0.6)
        axs[idx].set_xlabel("x")
        axs[idx].set_ylabel("μ(x)")

    # Eliminar subplots vacíos si los hay
    for j in range(idx + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Mostrar gráficas
plot_all_memberships(membership_PI, "Funciones de Membresía - Controlador PI Difuso")
plot_all_memberships(membership_D, "Funciones de Membresía - Controlador D Difuso")
