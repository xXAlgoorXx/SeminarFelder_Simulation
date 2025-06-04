import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import laplace, gaussian_filter

# ------------------ Optimierte Parameter für globale Struktur ------------------
a = 0.6
b = 1.0
epsilon = 0.005           # langsam: gut für Turing-Flecken
Du = 1e-3
Dv = 3e-3                 # ≈ 3× Du → größere Muster
dt = 0.05
N = 300                   # ausreichend groß für globales Muster
steps_per_frame = 10
dx = 1.0

# ------------------ Initialisierung: glatt + leicht gestört ------------------
np.random.seed(0)
u = np.ones((N, N)) * 0.2 + 0.005 * np.random.rand(N, N)
v = np.ones((N, N)) * 0.2 + 0.005 * np.random.rand(N, N)

# Optional: zentrale Störung zur Musterauslösung
r = 8
u = np.ones((N, N)) * 0.2 + 0.01 * (np.random.rand(N, N) - 0.5)
v = np.ones((N, N)) * 0.2 + 0.01 * (np.random.rand(N, N) - 0.5)


# ------------------ Reaktions-Diffusions-Schritt ------------------
def update():
    global u, v
    Lu = laplace(u, mode='wrap') / dx**2
    Lv = laplace(v, mode='wrap') / dx**2

    fu = u - (u**3) / 3 - v
    gv = epsilon * (u + a - b * v)

    u += dt * (Du * Lu + fu)
    v += dt * (Dv * Lv + gv)

# ------------------ Animation ------------------
fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
im = ax.imshow(u, cmap='inferno', interpolation='bicubic', vmin=-1.5, vmax=1.5)
ax.set_title("FitzHugh–Nagumo: großräumiges Turing-Muster")
ax.axis('off')

def animate(_):
    for _ in range(steps_per_frame):
        update()
    u_disp = gaussian_filter(u, sigma=0.8)  # Anzeige glätten
    im.set_data(u_disp)
    return [im]

ani = FuncAnimation(fig, animate, interval=50, blit=True)
plt.tight_layout()
plt.show()
