import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use("Qt5Agg")  # Falls nötig

# -------------------- Parameter --------------------
nx, ny = 200, 200
dt = 0.5                    # Zeitschritt

Du, Dv = 1,20         # Diffusion
e_v = 0.5
a_v = 1
a_z = -0.01

Du, Dv = 0.0005, 0.008
dt = 0.01
lambda_u = 0.9
sigma = 0.2
tau = 10
kappa = -0.01
# -------------------- Laplace-Kernel --------------------
laplace_kernel = np.array([[0.05, 0.2, 0.05],
                           [0.2, -1.0, 0.2],
                           [0.05, 0.2, 0.05]])
# Laplacian operator using periodic boundary conditions
def laplacian(Z):
    return (np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z)
# -------------------- Initialisierung --------------------
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
u = 0.1 * np.random.randn(ny, nx)
v = 0.1 * np.random.randn(ny, nx)

# # kleine zufällige Störungen
# u += 0.01 * np.random.rand(ny, nx)
# v += 0.01 * np.random.rand(ny, nx)

# zentrale Störung
r = 20
u[ny//2 - r:ny//2 + r, nx//2 - r:nx//2 + r] = 1.1
# v[ny//2 - r:ny//2 + r, nx//2 - r:nx//2 + r] = 0.25

# -------------------- Update-Funktion --------------------
def update(frame):
    global u, v
    # Lu = convolve2d(u, laplace_kernel, mode='same', boundary='wrap')
    # Lv = convolve2d(v, laplace_kernel, mode='same', boundary='wrap')
    Lu = laplacian(u)
    Lv = laplacian(v)

    fu = lambda_u * u - u**3 - kappa
    du = Du * laplacian(u) + fu - sigma * v
    dv = Dv * laplacian(v) + (u - v) / tau

    u += dt * du
    v += dt * dv
    # u += dt * (Du * Lu + u - u**3 - v)
    # v += dt * (Dv * Lv + e_v*(u-a_v*v-a_z))

    u[:] = np.clip(u, 0, 2)
    v[:] = np.clip(v, 0, 2)
    print(f"step={frame}")

    im.set_array(u)
    return [im]

# -------------------- Animation --------------------
fig, ax = plt.subplots()
im = ax.imshow(u, cmap='inferno', interpolation='bicubic')
ax.set_title("Fitzhugh Nagumo Turing-Muster")
ax.axis('off')

anim = FuncAnimation(fig, update, frames=1000, interval=30, blit=True)
plt.tight_layout()
plt.show()

# Optional speichern:
# anim.save("gray_scott.gif", fps=30)
