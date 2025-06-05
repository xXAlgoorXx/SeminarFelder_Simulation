import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
import sys
# Grid & time settings
size = 200
dt = 0.01
Du, Dv = 1.0, 20.0

# Reaktionsparameter mit Gleichgewicht u0 = v0 = 1
a, b = 1.0, -1.4
c, d = 1.0, -1.1
k1, k2 = 0.4, 0.1  # Konstanten für f(u,v) und g(u,v)

# Initial condition near u0 = v0 = 1
U = np.ones((size, size)) + 0.1 * np.random.randn(size, size)
V = np.ones((size, size)) + 0.1 * np.random.randn(size, size)

# Laplacian operator
def laplacian(Z):
    return (
        -4 * Z
        + np.roll(Z, (0, 1), (0, 1))
        + np.roll(Z, (0, -1), (0, 1))
        + np.roll(Z, (1, 0), (0, 1))
        + np.roll(Z, (-1, 0), (0, 1))
    )
i = 0
# Update rule
def update(frame):
    global U, V, i
    for _ in range(5):
        Lu = laplacian(U)
        Lv = laplacian(V)
        f = a * U + b * V + k1
        g = c * U + d * V + k2
        U += dt * (Du * Lu + f)
        V += dt * (Dv * Lv + g)
        U = np.clip(U, 0, 10)
        V = np.clip(V, 0, 10)
    print(f"Step {i}")
    if i == 1:
        plt.savefig("easy_n1.png", dpi=300,bbox_inches='tight', pad_inches=0)
    if i == 300:
        plt.savefig("easy_n300.png", dpi=300,bbox_inches='tight', pad_inches=0)
    if i == 999:
        plt.savefig("easy_n999.png", dpi=300,bbox_inches='tight', pad_inches=0)
        sys.exit(0)  # Stop after 1000 steps
    diff = gaussian_filter(U - V, sigma=1)
    norm_diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    im.set_array(norm_diff)
    i += 1
    return [im]

# Plot
fig, ax = plt.subplots()
im = ax.imshow(U - V, cmap='viridis', interpolation='bilinear', animated=True)
ax.axis('off')

ani = FuncAnimation(fig, update, frames=300, interval=30, blit=True)
plt.show()
