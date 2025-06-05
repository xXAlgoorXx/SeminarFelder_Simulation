import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from matplotlib.animation import FuncAnimation

# Parameter
nx, ny = 200, 200  # Gittergröße
# --- Parameter für Turing-Muster ---
Du, Dv = 0.0002, 0.02  # langsamer Aktivator
a, b = 0.2, 0.8
dt = 0.5

# --- Anfangszustand ---
# u = a + b + 0.01 * np.random.randn(nx, ny)
# v = b / (a + b)**2 + 0.01 * np.random.randn(nx, ny)
u = np.ones((nx, ny)) * (a + b)
v = np.ones((nx, ny)) * (b / (a + b)**2)
u[nx//2-5:nx//2+5, ny//2-5:ny//2+5] += 0.1  # zentrale Beule


# Reaktions-Diffusions-Schritt
def step(u, v):
    Lu = laplace(u)
    Lv = laplace(v)
    
    du = Du * Lu + a - u + u**2 * v
    dv = Dv * Lv + b - u**2 * v
    
    u += du * dt
    v += dv * dt
    
    # Werte begrenzen für Stabilität
    u = np.clip(u, 0, 3)
    v = np.clip(v, 0, 3)
    
    return u, v

# Plot Setup
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
im_u = ax1.imshow(u, cmap="viridis", vmin=0, vmax=2)
im_v = ax2.imshow(v, cmap="plasma", vmin=0, vmax=2)
ax1.set_title("Aktivator u")
ax2.set_title("Inhibitor v")
for ax in (ax1, ax2):
    ax.axis("off")

# Animation
def update(frame):
    global u, v
    u, v = step(u, v)
    im_u.set_array(u)
    im_v.set_array(v)
    return [im_u, im_v]

ani = FuncAnimation(fig, update, frames=1000, interval=30)
plt.tight_layout()
plt.show()
