import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops

ops.wipe()

# Parameters
n, m, k, h = 8, 345.6, 3.404e5, 3.0

# Model setup
ops.model("basic", "-ndm", 2, "-ndf", 3)

for i in range(n + 1):
    ops.node(i, 0.0, i * h)

ops.fix(0, 1, 1, 1)
for i in range(1, n + 1):
    ops.fix(i, 0, 1, 1)
    ops.mass(i, m, 1e-10, 1e-10)

# Elements (shear stiffness: k = 12EI/h³)
ops.geomTransf("Linear", 1)
E, I = 2.1e8, k * h**3 / (12 * 2.1e8)

for i in range(n):
    ops.element("elasticBeamColumn", i + 1, i, i + 1, 1.0, E, I, 1)

# Eigen analysis - use fullGenLapack for small systems needing all modes
eigenvalues = ops.eigen("-fullGenLapack", n)

periods = 2 * np.pi / np.sqrt(eigenvalues)

print("Mode Periods (OpenSeesPy):")
for i, T in enumerate(periods):
    print(f"  Mode {i + 1}: T = {T:.4f} s")

# Mode shapes
mode_shapes = np.array(
    [
        [ops.nodeEigenvector(f, mode + 1, 1) for f in range(1, n + 1)]
        for mode in range(n)
    ]
).T
mode_shapes /= mode_shapes[-1, :]

# Plot
floors = np.arange(n + 1)
fig, axes = plt.subplots(2, 4, figsize=(14, 8))

for i, ax in enumerate(axes.flat):
    shape = np.insert(mode_shapes[:, i], 0, 0)
    ax.plot(shape, floors, "b-o", lw=2, ms=6)
    ax.axvline(0, color="k", ls="--", lw=0.5)
    ax.set(
        xlabel="Amplitude", ylabel="Floor", title=f"Mode {i + 1} (T={periods[i]:.3f}s)"
    )
    ax.set_ylim(0, n)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    "Mode Shapes of 8-Story Shear Building (OpenSeesPy)", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig("mode_shapes_opensees.png", dpi=150)
plt.show()

ops.wipe()
