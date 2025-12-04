"""
Modal Analysis: 8-Story Building with TMD (m_d = 0.02*M1)
Task 4, Part 1: Determine frequencies and mode shapes for doubled TMD mass.
"""

import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops

# =============================================================================
# PARAMETERS
# =============================================================================
NUM_STORIES = 8
MASS = 345.6  # tons (per floor)
STIFFNESS = 3.404e5  # kN/m (per story)
MASS_RATIO = 0.02  # m_d = 0.02 * M1
DAMPING_RATIO = 0.10  # TMD damping ratio (ξ_d = 10%)


# =============================================================================
# FUNCTIONS
# =============================================================================
def get_first_mode_properties():
    """Get ω1 and M1 from original 8-story building."""
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)

    for i in range(NUM_STORIES + 1):
        ops.node(i, 0.0)
    ops.fix(0, 1)

    for i in range(1, NUM_STORIES + 1):
        ops.mass(i, MASS)

    ops.uniaxialMaterial("Elastic", 1, STIFFNESS)
    for i in range(NUM_STORIES):
        ops.element("zeroLength", i + 1, i, i + 1, "-mat", 1, "-dir", 1)

    eigenvalues = ops.eigen("-fullGenLapack", NUM_STORIES)
    omega1 = np.sqrt(eigenvalues[0])

    # First mode shape normalized to roof = 1
    phi1 = np.array([ops.nodeEigenvector(i, 1, 1) for i in range(1, NUM_STORIES + 1)])
    phi1 /= phi1[-1]

    # Generalized mass: M1 = Σ m_i * φ_i²
    M1 = MASS * np.sum(phi1**2)

    return omega1, M1, phi1


def build_model_with_tmd(m_d, k_d):
    """Build 9-DOF system (8 floors + TMD) for modal analysis."""
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)

    # Nodes: 0=base, 1-8=floors, 9=TMD
    for i in range(NUM_STORIES + 2):
        ops.node(i, 0.0)
    ops.fix(0, 1)

    # Mass assignment
    for i in range(1, NUM_STORIES + 1):
        ops.mass(i, MASS)
    ops.mass(NUM_STORIES + 1, m_d)

    # Materials
    ops.uniaxialMaterial("Elastic", 1, STIFFNESS)
    ops.uniaxialMaterial("Elastic", 2, k_d)

    # Story elements (1-8)
    for i in range(NUM_STORIES):
        ops.element("zeroLength", i + 1, i, i + 1, "-mat", 1, "-dir", 1)

    # TMD element (roof to TMD)
    ops.element(
        "zeroLength",
        NUM_STORIES + 1,
        NUM_STORIES,
        NUM_STORIES + 1,
        "-mat",
        2,
        "-dir",
        1,
    )


def run_modal_analysis():
    """Extract periods, frequencies, and mode shapes."""
    num_dofs = NUM_STORIES + 1  # 8 floors + 1 TMD
    eigenvalues = ops.eigen("-fullGenLapack", num_dofs)

    omega = np.sqrt(eigenvalues)
    frequencies = omega / (2 * np.pi)
    periods = 1 / frequencies

    # Mode shapes: rows = DOFs (floors 1-8, TMD), cols = modes
    mode_shapes = np.array(
        [
            [ops.nodeEigenvector(node, mode + 1, 1) for node in range(1, num_dofs + 1)]
            for mode in range(num_dofs)
        ]
    ).T

    # Normalize to roof (node 8) = 1
    mode_shapes /= mode_shapes[NUM_STORIES - 1, :]

    return periods, frequencies, mode_shapes


def plot_mode_shapes(periods, mode_shapes):
    """Plot all 9 mode shapes."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    for i, ax in enumerate(axes.flat):
        shape = np.insert(mode_shapes[:, i], 0, 0)  # Add zero at base
        floors = np.arange(len(shape))

        ax.plot(shape[:9], floors[:9], "b-o", lw=2, ms=6, label="Building")
        ax.plot(shape[9], floors[9], "rs", ms=10, label="TMD")
        ax.axvline(0, color="k", ls="--", lw=0.5)
        ax.set_xlabel("Amplitude")
        ax.set_ylabel("Floor / TMD")
        ax.set_title(f"Mode {i + 1} (T = {periods[i]:.4f} s)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="lower right")

    plt.suptitle(
        r"Mode Shapes: 8-Story Building + TMD $m_d = 0.02 M_1$ (OpenSees)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Step 1: Get first mode properties from original building
    omega1, M1, phi1 = get_first_mode_properties()
    T1 = 2 * np.pi / omega1

    print("=" * 60)
    print("ORIGINAL BUILDING - First Mode Properties")
    print("=" * 60)
    print(f"  Period T1        = {T1:.4f} s")
    print(f"  Frequency ω1     = {omega1:.4f} rad/s")
    print(f"  Generalized M1   = {M1:.2f} tons")

    # Step 2: Calculate TMD parameters
    m_d = MASS_RATIO * M1
    omega_d = omega1  # Tuned to first mode
    k_d = m_d * omega_d**2
    c_d = 2 * DAMPING_RATIO * m_d * omega_d

    print("\n" + "=" * 60)
    print("TMD PARAMETERS (m_d = 0.02 × M1)")
    print("=" * 60)
    print(f"  TMD mass m_d     = {m_d:.2f} tons")
    print(f"  TMD stiffness k_d = {k_d:.2f} kN/m")
    print(f"  TMD damping c_d  = {c_d:.2f} kN·s/m")
    print(f"  TMD frequency ω_d = {omega_d:.4f} rad/s")

    # Step 3: Build model and run modal analysis
    build_model_with_tmd(m_d, k_d)
    periods, frequencies, mode_shapes = run_modal_analysis()

    print("\n" + "=" * 60)
    print("MODAL PROPERTIES - Building + TMD System (9 DOFs)")
    print("=" * 60)
    print(f"  {'Mode':<6} {'Period (s)':<14} {'Frequency (Hz)':<16} {'ω (rad/s)':<12}")
    print("  " + "-" * 48)
    for i in range(len(periods)):
        print(
            f"  {i + 1:<6} {periods[i]:<14.4f} {frequencies[i]:<16.4f} {2 * np.pi * frequencies[i]:<12.4f}"
        )

    # Step 4: Plot and save
    fig = plot_mode_shapes(periods, mode_shapes)
    fig.savefig("mode_shapes_tmd_002--opensees.png", dpi=150)
    plt.show()

    ops.wipe()


if __name__ == "__main__":
    main()
