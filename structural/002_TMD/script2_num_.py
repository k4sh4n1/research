import matplotlib.pyplot as plt
import numpy as np


def build_matrices(n, m, k):
    """Build mass and stiffness matrices for n-story shear building."""
    M = np.diag([m] * n)
    K = np.diag([2 * k] * n) + np.diag([-k] * (n - 1), 1) + np.diag([-k] * (n - 1), -1)
    K[-1, -1] = k
    return M, K


def modal_analysis(M, K):
    """Return natural frequencies (rad/s) and normalized mode shapes."""
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(M) @ K)
    idx = np.argsort(eigvals.real)
    omega = np.sqrt(eigvals[idx].real)
    phi = eigvecs[:, idx].real
    phi /= phi[-2, :]  # normalize roof to 1
    return omega, phi


def add_tmd(M, K, m_d, omega_d):
    """Extend system matrices to include TMD on roof."""
    n = M.shape[0]
    k_d = m_d * omega_d**2

    M_tmd = np.zeros((n + 1, n + 1))
    M_tmd[:n, :n] = M
    M_tmd[n, n] = m_d

    K_tmd = np.zeros((n + 1, n + 1))
    K_tmd[:n, :n] = K
    K_tmd[n - 1, n - 1] += k_d
    K_tmd[n - 1, n] = K_tmd[n, n - 1] = -k_d
    K_tmd[n, n] = k_d

    return M_tmd, K_tmd


def plot_mode_shapes(omega, phi, title, filename):
    """Plot all mode shapes in grid layout."""
    n_modes = len(omega)
    periods = 2 * np.pi / omega
    floors = np.arange(n_modes + 1)

    cols = 3
    rows = (n_modes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows))

    for i, ax in enumerate(axes.flat):
        if i >= n_modes:
            ax.axis("off")
            continue
        shape = np.insert(phi[:, i], 0, 0)
        ax.plot(shape, floors, "b-o", lw=2, ms=5)
        ax.axvline(0, color="k", ls="--", lw=0.5)
        ax.set_xlabel("Amplitude")
        ax.set_ylabel("Floor")
        ax.set_title(f"Mode {i + 1} (T = {periods[i]:.3f} s)")
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    return fig


def main():
    # Parameters
    N = 8  # number of stories
    MASS = 345.6  # tons
    STIFFNESS = 3.404e5  # kN/m
    MASS_RATIO = 0.02  # TMD mass ratio (doubled from 0.01)

    # Original system
    M, K = build_matrices(N, MASS, STIFFNESS)
    omega, phi = modal_analysis(M, K)

    omega_1 = omega[0]
    phi_1 = phi[:, 0]
    M_1 = phi_1 @ M @ phi_1  # generalized mass

    print("=" * 55)
    print("ORIGINAL 8-STORY SYSTEM")
    print("=" * 55)
    print(f"  ω₁ = {omega_1:.4f} rad/s")
    print(f"  T₁ = {2 * np.pi / omega_1:.4f} s")
    print(f"  M₁ = {M_1:.2f} tons")

    # TMD parameters
    m_d = MASS_RATIO * M_1
    omega_d = omega_1
    k_d = m_d * omega_d**2

    print("\n" + "=" * 55)
    print(f"TMD PARAMETERS (mass ratio = {MASS_RATIO})")
    print("=" * 55)
    print(f"  m_d = {MASS_RATIO} × M₁ = {m_d:.2f} tons")
    print(f"  ω_d = ω₁ = {omega_d:.4f} rad/s")
    print(f"  k_d = {k_d:.2f} kN/m")

    # System with TMD
    M_tmd, K_tmd = add_tmd(M, K, m_d, omega_d)
    omega_tmd, phi_tmd = modal_analysis(M_tmd, K_tmd)

    print("\n" + "=" * 55)
    print("SYSTEM WITH TMD (9 DOF)")
    print("=" * 55)
    print(f"{'Mode':<6} {'ω (rad/s)':<14} {'T (s)':<10}")
    print("-" * 30)
    for i, w in enumerate(omega_tmd):
        print(f"{i + 1:<6} {w:<14.4f} {2 * np.pi / w:<10.4f}")

    # Frequency split comparison
    print("\n" + "=" * 55)
    print("FREQUENCY COMPARISON")
    print("=" * 55)
    print(f"  Original ω₁:     {omega_1:.4f} rad/s")
    print(f"  With TMD ω₁:     {omega_tmd[0]:.4f} rad/s")
    print(f"  With TMD ω₂:     {omega_tmd[1]:.4f} rad/s")
    print(f"  Frequency split: {omega_tmd[1] - omega_tmd[0]:.4f} rad/s")

    # Plot
    plot_mode_shapes(
        omega_tmd,
        phi_tmd,
        f"Mode Shapes: 8-Story Building + TMD $m_d = {MASS_RATIO} M_1$ (NumPy)",
        "mode_shapes_tmd_002.png",
    )
    plt.show()


if __name__ == "__main__":
    main()
