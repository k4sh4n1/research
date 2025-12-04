import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops


def create_building_with_tmd(
    num_stories, mass, stiffness, damping, tmd_mass, tmd_stiffness, tmd_damping
):
    """Create 8-story shear building with TMD on roof."""
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)

    # Nodes: 0=base, 1-8=floors, 9=TMD
    for i in range(num_stories + 2):
        ops.node(i, float(i))

    ops.fix(0, 1)

    # Mass: floors 1-8 + TMD at node 9
    for i in range(1, num_stories + 1):
        ops.mass(i, mass)
    ops.mass(num_stories + 1, tmd_mass)

    # Materials: story stiffness + TMD stiffness
    ops.uniaxialMaterial("Elastic", 1, stiffness)
    ops.uniaxialMaterial("Elastic", 2, tmd_stiffness)

    # Elements: 8 story springs + 1 TMD spring
    for i in range(num_stories):
        ops.element("zeroLength", i + 1, i, i + 1, "-mat", 1, "-dir", 1)

    # TMD connected to roof (node 8 to node 9)
    ops.element(
        "zeroLength",
        num_stories + 1,
        num_stories,
        num_stories + 1,
        "-mat",
        2,
        "-dir",
        1,
    )


def run_modal_analysis(num_dofs):
    """Run eigenvalue analysis and return periods and mode shapes."""
    eigenvalues = ops.eigen("-fullGenLapack", num_dofs)
    periods = 2 * np.pi / np.sqrt(eigenvalues)
    frequencies = np.sqrt(eigenvalues) / (2 * np.pi)

    mode_shapes = np.array(
        [
            [ops.nodeEigenvector(node, mode + 1, 1) for node in range(1, num_dofs + 1)]
            for mode in range(num_dofs)
        ]
    ).T

    # Normalize to max absolute value = 1
    for i in range(num_dofs):
        mode_shapes[:, i] /= np.max(np.abs(mode_shapes[:, i]))

    return periods, frequencies, mode_shapes


def calculate_tmd_parameters(
    mass, mode_shape_roof, omega1, mass_ratio=0.01, freq_ratio=1.0, damping_ratio=0.10
):
    """Calculate TMD parameters based on first mode properties."""
    # Generalized mass of first mode: M1 = sum(m_i * phi_i^2)
    M1 = mass * np.sum(mode_shape_roof**2)

    m_d = mass_ratio * M1
    omega_d = freq_ratio * omega1
    k_d = m_d * omega_d**2
    c_d = 2 * damping_ratio * m_d * omega_d

    print(f"\n--- TMD Parameters ---")
    print(f"Generalized mass M1 = {M1:.2f} tons")
    print(f"TMD mass (m_d = 0.01*M1) = {m_d:.2f} tons")
    print(f"TMD stiffness k_d = {k_d:.2f} kN/m")
    print(f"TMD damping c_d = {c_d:.2f} tons/s")

    return m_d, k_d, c_d, M1


def plot_mode_shapes(periods, mode_shapes, title, filename):
    """Plot first 9 mode shapes."""
    num_modes = mode_shapes.shape[1]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    for i, ax in enumerate(axes.flat):
        shape = np.insert(mode_shapes[:, i], 0, 0)  # Add zero at base
        floors = np.arange(len(shape))

        ax.plot(shape[:9], floors[:9], "b-o", lw=2, ms=6, label="Building")
        ax.plot(shape[9], floors[9], "rs", ms=10, label="TMD")
        ax.axvline(0, color="k", ls="--", lw=0.5)
        ax.set_xlabel("Amplitude")
        ax.set_ylabel("Floor / TMD")
        ax.set_title(f"Mode {i + 1} (T={periods[i]:.4f}s)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="lower right")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    return fig


def main():
    # Building parameters
    NUM_STORIES = 8
    MASS = 345.6  # tons
    STIFFNESS = 3.404e5  # kN/m
    DAMPING = 2937.0  # tons/s

    # --- Step 1: Get first mode properties from original building ---
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    for i in range(NUM_STORIES + 1):
        ops.node(i, float(i))
    ops.fix(0, 1)
    for i in range(1, NUM_STORIES + 1):
        ops.mass(i, MASS)
    ops.uniaxialMaterial("Elastic", 1, STIFFNESS)
    for i in range(NUM_STORIES):
        ops.element("zeroLength", i + 1, i, i + 1, "-mat", 1, "-dir", 1)

    eigenvalues_orig = ops.eigen("-fullGenLapack", NUM_STORIES)
    omega1 = np.sqrt(eigenvalues_orig[0])
    T1 = 2 * np.pi / omega1

    mode_shape_1 = np.array(
        [ops.nodeEigenvector(i, 1, 1) for i in range(1, NUM_STORIES + 1)]
    )
    mode_shape_1 /= mode_shape_1[-1]  # Normalize to roof = 1

    print(f"Original building first mode: T1 = {T1:.4f} s, ω1 = {omega1:.4f} rad/s")

    # --- Step 2: Calculate TMD parameters ---
    m_d, k_d, c_d, M1 = calculate_tmd_parameters(MASS, mode_shape_1, omega1)

    # --- Step 3: Create building with TMD and run modal analysis ---
    create_building_with_tmd(NUM_STORIES, MASS, STIFFNESS, DAMPING, m_d, k_d, c_d)
    periods, frequencies, mode_shapes = run_modal_analysis(NUM_STORIES + 1)

    # Print results
    print(f"\n--- Modal Periods (Building + TMD) ---")
    for i, T in enumerate(periods, 1):
        print(f"  Mode {i}: T = {T:.4f} s, f = {frequencies[i - 1]:.4f} Hz")

    # Plot and save
    fig = plot_mode_shapes(
        periods,
        mode_shapes,
        "Mode Shapes of 8-Story Building + TMD (OpenSees)",
        "mode_shapes_with_tmd--opensees.png",
    )
    plt.show()

    ops.wipe()


if __name__ == "__main__":
    main()
