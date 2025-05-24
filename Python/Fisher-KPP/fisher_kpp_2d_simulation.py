import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams.update({
    "text.usetex": True,              # LaTeX verwenden
    "font.family": "serif",           # Serifen-Schrift (wie in LaTeX)
    "font.serif": ["Computer Modern"],# Klassische LaTeX-Schrift
    "axes.labelsize": 12,             # Achsenbeschriftung
    "font.size": 12,                  # allgemeine Schriftgröße
})

# Parameter
D = 0.1    # Diffusionskoeffizient
r = 1.0    # Wachstumsrate
L = 50     # Länge des quadratischen Gebiets
Nx = 100   # Gitterpunkte pro Dimension
dx = L / Nx
dt = 0.1   # Zeitschritt
T = 50     # Gesamtzeit
Nt = int(T / dt)

# Gitter
x = np.linspace(-L/2, L/2, Nx)
y = np.linspace(-L/2, L/2, Nx)
X, Y = np.meshgrid(x, y)

# Anfangsbedingung: runder Fleck in der Mitte
u = np.zeros((Nx, Nx))
u0 = np.exp(-((X**2 + Y**2)/10))  # glatte Besiedlung im Zentrum
u = u0.copy()

# Hilfsfunktion: Laplace-Operator (finite Differenzen)
def laplacian(U):
    return (
        np.roll(U, 1, axis=0) + np.roll(U, -1, axis=0) +
        np.roll(U, 1, axis=1) + np.roll(U, -1, axis=1) -
        4 * U
    ) / dx**2

# Speicherung für Animation
frames = []
save_every = 20  # speichere alle n Schritte


# Zwei Momentaufnahmen speichern
snapshots = {}
for n in range(Nt):
    Lu = laplacian(u)
    u += dt * (D * Lu + r * u * (1 - u))
    if n % save_every == 0:
        frames.append(u.copy())
    if n == 0:
        snapshots["t0"] = u.copy()
    elif n == Nt // 2:
        snapshots["t1"] = u.copy()

# Animation erstellen
fig, ax = plt.subplots(figsize=(6, 5))
img = ax.imshow(frames[0], extent=(-L/2, L/2, -L/2, L/2), origin='lower', cmap='viridis', vmin=0, vmax=1)
ax.set_title("$Fisher-KPP in 2D$")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

def update(frame):
    img.set_data(frame)
    return [img]

anim = FuncAnimation(fig, update, frames=frames, interval=100)
anim.save("fisher_kpp_2d.gif", writer="pillow", fps=10)

plt.show()

# Plot der beiden Zeitpunkte
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (label, data) in zip(axes, snapshots.items()):
    img = ax.imshow(data, extent=(-L/2, L/2, -L/2, L/2), origin='lower',
                    cmap='viridis', vmin=0, vmax=1)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

# Farbleiste und Layout
# fig.colorbar(img)
# fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.7, label="u(x, y)")
plt.tight_layout()
plt.savefig("fisher_kpp_2d_wave_comparison.pdf", bbox_inches='tight')
plt.show()
