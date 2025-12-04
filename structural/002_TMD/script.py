import numpy as np

# Given parameters
n = 8  # number of stories
m = 345.6  # tons (same for all floors)
k = 3.404e5  # kN/m (same for all floors)

# Mass matrix (diagonal)
M = np.diag([m] * n)

# Stiffness matrix (tridiagonal for shear building)
K = np.zeros((n, n))
for i in range(n):
    if i == 0:
        K[i, i] = k + k
        K[i, i + 1] = -k
    elif i == n - 1:
        K[i, i] = k
        K[i, i - 1] = -k
    else:
        K[i, i] = k + k
        K[i, i - 1] = -k
        K[i, i + 1] = -k

# Solve eigenvalue problem: K*phi = omega^2*M*phi
eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(M) @ K)

# Sort by frequency
idx = np.argsort(eigenvalues)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Natural frequencies
omega = np.sqrt(eigenvalues)  # rad/s
freq = omega / (2 * np.pi)  # Hz
period = 1 / freq  # seconds

# Normalize mode shapes (roof = 1)
mode_shapes = eigenvectors / eigenvectors[-1, :]

# Print results
print("=" * 50)
print("MODAL ANALYSIS RESULTS")
print("=" * 50)
print(f"\n{'Mode':<6}{'ω (rad/s)':<14}{'f (Hz)':<12}{'T (s)':<10}")
print("-" * 42)
for i in range(n):
    print(f"{i + 1:<6}{omega[i]:<14.4f}{freq[i]:<12.4f}{period[i]:<10.4f}")

print("\n" + "=" * 50)
print("MODE SHAPES (normalized to roof = 1)")
print("=" * 50)
print(f"\n{'Floor':<8}", end="")
for i in range(n):
    print(f"Mode {i + 1:<6}", end="")
print("\n" + "-" * 80)
for floor in range(n):
    print(f"{floor + 1:<8}", end="")
    for mode in range(n):
        print(f"{mode_shapes[floor, mode]:<10.4f}", end="")
    print()
