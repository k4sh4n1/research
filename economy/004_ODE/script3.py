import matplotlib.pyplot as plt
import numpy as np

# Create grid
t = np.linspace(-2, 2, 20)
y = np.linspace(-3, 3, 20)
T, Y = np.meshgrid(t, y)

# dy/dt = y (slope at each point)
dT = np.ones_like(T)  # dt component = 1 (moving forward in time)
dY = Y  # dy component = y

# Normalize for consistent arrow lengths
magnitude = np.sqrt(dT**2 + dY**2)
dT_norm = dT / magnitude
dY_norm = dY / magnitude

# Plot
plt.figure(figsize=(10, 6))
plt.quiver(T, Y, dT_norm, dY_norm, alpha=0.7)
plt.xlabel("$t$")
plt.ylabel("$y$")
plt.title(r"Direction Field for $\frac{dy}{dt} = y$")
plt.axhline(y=0, color="red", linestyle="--", label="Equilibrium $y=0$")
plt.legend()
plt.grid(True)
plt.show()
