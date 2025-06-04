import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import convolve2d
import matplotlib
matplotlib.use("TkAgg")  # oder "Qt5Agg", falls Qt installiert ist

# Parameter: bewährte Werte
nx, ny = 300, 300

dt = 0.2
steps_per_frame = 10
frames = 500

a = 0.1
b = 0.7
Du = 0.00016
Dv = 0.008

# Laplace-Operator
laplace_kernel = np.array([[0.05, 0.2, 0.05],
                           [0.2, -1.0, 0.2],
                           [0.05, 0.2, 0.05]])

# Initialzustand mit starkem Rauschen
np.random.seed(42)
u = np.ones((ny, nx)) + 0.01 * (np.random.rand(ny, nx) - 0.5)
v = np.ones((ny, nx)) + 0.01 * (np.random.rand(ny, nx) - 0.5)


# Anzeige
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(u, cmap='inferno', interpolation='bicubic', vmin=0.8, vmax=1.2)
ax.set_title("Schnakenberg: Turing-Muster")
ax.axis('off')

# Update-Funktion
def update(frame):
    global u, v
    for _ in range(steps_per_frame):
        Lu = convolve2d(u, laplace_kernel, mode='same', boundary='wrap')
        Lv = convolve2d(v, laplace_kernel, mode='same', boundary='wrap')
        uv2 = u * u * v
        u += dt * (Du * Lu + a - u + uv2)
        v += dt * (Dv * Lv + b - uv2)
        u = np.clip(u, 0, 5)
        v = np.clip(v, 0, 5)
    im.set_array(u)
    im.set_clim(vmin=np.min(u), vmax=np.max(u))
    return [im]

# Animation starten
ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
plt.tight_layout()
plt.show()
