import matplotlib.pyplot as plt
import numpy as np


def build_matrices(num_stories, mass, stiffness):
    """Construct mass and stiffness matrices for shear building."""
    M = np.diag([mass] * num_stories)

    K = np.diag([2 * stiffness] * num_stories)
    K += np.diag([-stiffness] * (num_stories - 1), 1)
    K += np.diag([-stiffness] * (num_stories - 1), -1)
    K[-1, -1] = stiffness

    return M, K


def modal_analysis(M, K):
    """Solve eigenvalue problem and return frequencies and mode shapes."""
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(M) @ K)

    idx = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[idx].real
    eigenvectors = eigenvectors[:, idx].real

    frequencies = np.sqrt(eigenvalues)  # rad/s
    mode_shapes = eigenvectors / eigenvectors[-1, :]  # normalize roof to 1

    return frequencies, mode_shapes


def add_tmd(M, K, m_d, omega_d):
    """Extend system matrices to include TMD on roof."""
    n = M.shape[0]
    k_d = m_d * omega_d**2

    # Extend mass matrix
    M_tmd = np.zeros((n + 1, n + 1))
    M_tmd[:n, :n] = M
    M_tmd[n, n] = m_d

    # Extend stiffness matrix
    K_tmd = np.zeros((n + 1, n + 1))
    K_tmd[:n, :n] = K
    K_tmd[n - 1, n - 1] += k_d  # add TMD stiffness to roof
    K_tmd[n - 1, n] = -k_d
    K_tmd[n, n - 1] = -k_d
    K_tmd[n, n] = k_d

    return M_tmd, K_tmd


def compute_generalized_mass(M, phi):
    """Compute generalized mass: M_i = phi_i^T @ M @ phi_i"""
    return phi.T @ M @ phi


def plot_mode_shapes(frequencies, mode_shapes, title, filename):
    """Plot mode shapes in a grid layout."""
    num_modes = len(frequencies)
    periods = 2 * np.pi / frequencies
    floors = np.arange(num_modes + 1)

    rows = 2 if num_modes <= 8 else 3
    cols = (num_modes + rows - 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 4 * rows))

    for i, ax in enumerate(axes.flat):
        if i >= num_modes:
            ax.axis("off")
            continue

        shape = np.insert(mode_shapes[:, i], 0, 0)
        ax.plot(shape, floors, "b-o", lw=2, ms=6)
        ax.axvline(0, color="k", ls="--", lw=0.5)
        ax.set_xlabel("Amplitude")
        ax.set_ylabel("Floor")
        ax.set_title(f"Mode {i + 1} (T={periods[i]:.3f}s)")
        ax.set_ylim(0, num_modes)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    return fig


def main():
    # Building parameters
    NUM_STORIES = 8
    MASS = 345.6  # tons
    STIFFNESS = 3.404e5  # kN/m
    ZETA_D = 0.10  # TMD damping ratio (for Task 3)

    # Step 1: Original system modal analysis
    M, K = build_matrices(NUM_STORIES, MASS, STIFFNESS)
    omega, phi = modal_analysis(M, K)

    omega_1 = omega[0]
    phi_1 = phi[:, 0]
    M_1 = compute_generalized_mass(M, phi_1)

    print("=" * 50)
    print("ORIGINAL 8-STORY SYSTEM")
    print("=" * 50)
    print(f"First mode frequency: ω₁ = {omega_1:.4f} rad/s")
    print(f"First mode period:    T₁ = {2 * np.pi / omega_1:.4f} s")
    print(f"Generalized mass M₁:  {M_1:.2f} tons")

    # Step 2: TMD parameters
    m_d = 0.01 * M_1
    omega_d = omega_1
    k_d = m_d * omega_d**2

    print("\n" + "=" * 50)
    print("TMD PARAMETERS")
    print("=" * 50)
    print(f"TMD mass:      m_d = 0.01 × M₁ = {m_d:.2f} tons")
    print(f"TMD frequency: ω_d = ω₁ = {omega_d:.4f} rad/s")
    print(f"TMD stiffness: k_d = {k_d:.2f} kN/m")
    print(f"TMD damping:   ζ_d = {ZETA_D * 100:.0f}%")

    # Step 3: System with TMD
    M_tmd, K_tmd = add_tmd(M, K, m_d, omega_d)
    omega_tmd, phi_tmd = modal_analysis(M_tmd, K_tmd)

    print("\n" + "=" * 50)
    print("SYSTEM WITH TMD (9 DOF)")
    print("=" * 50)
    print("\nModal Frequencies and Periods:")
    print("-" * 35)
    for i, w in enumerate(omega_tmd):
        T = 2 * np.pi / w
        print(f"  Mode {i + 1}: ω = {w:8.4f} rad/s, T = {T:.4f} s")

    # Step 4: Plot mode shapes
    plot_mode_shapes(
        omega_tmd,
        phi_tmd,
        "Mode Shapes of 8-Story Building with TMD (NumPy)",
        "mode_shapes_tmd.png",
    )

    # Compare first two modes (split due to TMD)
    print("\n" + "=" * 50)
    print("FREQUENCY COMPARISON")
    print("=" * 50)
    print(f"Original ω₁:     {omega_1:.4f} rad/s")
    print(f"With TMD ω₁:     {omega_tmd[0]:.4f} rad/s")
    print(f"With TMD ω₂:     {omega_tmd[1]:.4f} rad/s")
    print(f"Frequency split: {(omega_tmd[1] - omega_tmd[0]):.4f} rad/s")

    plt.show()


if __name__ == "__main__":
    main()
