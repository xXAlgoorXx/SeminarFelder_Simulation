import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage import gaussian_filter
import sys

# Laplacian operator using periodic boundary conditions
def laplacian(Z):
    return (np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z)



# Grid and simulation settings
nx, ny = 200, 200

# # Klein Leopard Turing Patterns
# Du, Dv = 0.1, 23
# lambda_u = 0.9 #U
# sigma = 0.3 #U
# tau = 15 #V
# kappa = 0.004 #V

Du = 0.0005     # langsamer Aktivator
Dv = 0.015      # deutlich schnellerer Inhibitor

lambda_u = 0.5
sigma = 0.3

tau = 30        # langsam reagierender Inhibitor
kappa = 0.05    # schwache Hintergrundhemmung

dt = 0.01       # Zeitschritt

Du = 0.008
Dv = 0.04
lambda_u = 0.5
sigma = 0.3
tau = 30
kappa = 0.05


stoerkoeff = 0.8

# FitzHugh-Nagumo update
def updateFN(U, V, Du, Dv, dt):
    

    f_u = lambda_u * U - U**3 - kappa

    Lu = laplacian(U)
    Lv = laplacian(V)

    dU = Du * Lu + f_u - sigma * V
    dV = Dv * Lv + (U - V) / tau

    U += dU * dt
    V += dV * dt

    clipValue = 3
    # Clip to avoid explosion
    np.clip(U, -clipValue, clipValue, out=U)
    np.clip(V, -clipValue, clipValue, out=V)

    return U, V

# Smoothed random initial conditions
U = stoerkoeff * np.random.randn(nx, ny)
V = stoerkoeff * np.random.randn(nx, ny)
U = gaussian_filter(U, sigma=3)
V = gaussian_filter(V, sigma=3)

# # Zentrale Störung
# U = np.zeros((nx, ny))
# V = np.zeros((nx, ny))
# U[nx//2-3:nx//2+3, ny//2-3:ny//2+3] = 0.5  # kleine zentrale Störung

# Setup animation
fig, ax = plt.subplots()
im = ax.imshow(U, cmap="coolwarm", animated=True, interpolation="nearest", vmin=-1, vmax=1)

def animate(i):
    global U, V
    U, V = updateFN(U, V, Du, Dv, dt)
    im.set_array(U)
    # print(f"Step {i}")
    # if i == 1:
    #     plt.savefig("fhn_n1.png", dpi=300,bbox_inches='tight', pad_inches=0)
    # if i == 300:
    #     plt.savefig("fhn_n300.png", dpi=300,bbox_inches='tight', pad_inches=0)
    # if i == 999:
    #     plt.savefig("fhn_n999.png", dpi=300,bbox_inches='tight', pad_inches=0)
    #     sys.exit(0)  # Stop after 1000 steps
    return [im]

anim = animation.FuncAnimation(fig, animate,interval=5, blit=False)
dvdu = Dv/Du
# plt.title(f"FitzHugh-Nagumo Turing Patterns({dvdu:.2f})")

plt.axis("off")

plt.show()

# Save animation (optional)
anim.save('fhn_large_patterns.gif',fps = 40)


