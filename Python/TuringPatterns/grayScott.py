import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use("Qt5Agg")  # Falls nötig

# -------------------- Parameter --------------------
nx, ny = 200, 200
Du, Dv = 0.16, 0.08         # Diffusion
F, k = 0.035, 0.06         # Reaktionsraten
dt = 5.0                    # Zeitschritt

# -------------------- Laplace-Kernel --------------------
laplace_kernel = np.array([[0.05, 0.2, 0.05],
                           [0.2, -1.0, 0.2],
                           [0.05, 0.2, 0.05]])

# -------------------- Initialisierung --------------------
u = np.ones((ny, nx))
v = np.zeros((ny, nx))

# kleine zufällige Störungen
u += 0.01 * np.random.rand(ny, nx)
v += 0.01 * np.random.rand(ny, nx)

# zentrale Störung
r = 20
u[ny//2 - r:ny//2 + r, nx//2 - r:nx//2 + r] = 0.50
v[ny//2 - r:ny//2 + r, nx//2 - r:nx//2 + r] = 0.25

# -------------------- Update-Funktion --------------------
def update(frame):
    global u, v
    Lu = convolve2d(u, laplace_kernel, mode='same', boundary='wrap')
    Lv = convolve2d(v, laplace_kernel, mode='same', boundary='wrap')

    uvv = u * v * v
    u += dt * (Du * Lu - uvv + F * (1 - u))
    v += dt * (Dv * Lv + uvv - (F + k) * v)

    u[:] = np.clip(u, 0, 1)
    v[:] = np.clip(v, 0, 1)
    print(f"step={frame}")

    im.set_array(u)
    if frame == 1:
        plt.savefig("gs_n1.png", dpi=300)
    if frame == 300:
        plt.savefig("gs_n300.png", dpi=300)
    if frame == 999:
        plt.savefig("gs_n999.png", dpi=300)
    return [im]

# -------------------- Animation --------------------
fig, ax = plt.subplots()
im = ax.imshow(u, cmap='inferno', interpolation='bicubic')
ax.set_title("Gray-Scott Turing-Muster")
ax.axis('off')

anim = FuncAnimation(fig, update, frames=1000, interval=30, blit=True)
plt.tight_layout()
plt.show()

# Optional speichern:
# anim.save("gray_scott.gif", fps=30)
