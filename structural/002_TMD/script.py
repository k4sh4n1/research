import matplotlib.pyplot as plt
import numpy as np

# Given parameters
n = 8
m = 345.6  # tons
k = 3.404e5  # kN/m

# Mass and stiffness matrices
M = np.diag([m] * n)
K = np.diag([2 * k] * n) + np.diag([-k] * (n - 1), 1) + np.diag([-k] * (n - 1), -1)
K[-1, -1] = k

# Solve eigenvalue problem
eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(M) @ K)
idx = np.argsort(eigenvalues)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Calculate frequencies and periods
omega = np.sqrt(eigenvalues)  # rad/s
periods = 2 * np.pi / omega  # seconds

# Log periods
print("Mode Periods:")
for i in range(n):
    print(f"  Mode {i + 1}: T = {periods[i]:.4f} s")

# Normalize mode shapes (roof = 1)
mode_shapes = eigenvectors / eigenvectors[-1, :]

# Floor levels (0 = ground, 1-8 = floors)
floors = np.arange(0, n + 1)

# Plot
fig, axes = plt.subplots(2, 4, figsize=(14, 8))
axes = axes.flatten()

for i in range(n):
    ax = axes[i]
    shape = np.insert(mode_shapes[:, i], 0, 0)

    ax.plot(shape, floors, "b-o", linewidth=2, markersize=6)
    ax.axvline(x=0, color="k", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Floor")
    ax.set_title(f"Mode {i + 1} (T={periods[i]:.3f}s)")
    ax.set_ylim(0, n)
    ax.set_yticks(range(n + 1))
    ax.grid(True, alpha=0.3)

plt.suptitle("Mode Shapes of 8-Story Shear Building", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("mode_shapes.png", dpi=150)
plt.show()
