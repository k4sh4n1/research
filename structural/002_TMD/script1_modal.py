import matplotlib.pyplot as plt
import numpy as np


def build_matrices(num_stories, mass, stiffness):
    """Construct mass and stiffness matrices for shear building."""
    M = np.diag([mass] * num_stories)

    K = np.diag([2 * stiffness] * num_stories)
    K += np.diag([-stiffness] * (num_stories - 1), 1)
    K += np.diag([-stiffness] * (num_stories - 1), -1)
    K[-1, -1] = stiffness  # Top floor has only one spring

    return M, K


def modal_analysis(M, K):
    """Solve eigenvalue problem and return periods and normalized mode shapes."""
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(M) @ K)

    # Sort by ascending eigenvalue (lowest frequency first)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Calculate periods and normalize mode shapes (roof = 1)
    periods = 2 * np.pi / np.sqrt(eigenvalues)
    mode_shapes = eigenvectors / eigenvectors[-1, :]

    return periods, mode_shapes


def plot_mode_shapes(
    periods, mode_shapes, title="Mode Shapes of 8-Story Shear Building (NumPy)"
):
    """Plot all mode shapes in a 2x4 grid."""
    num_modes = len(periods)
    floors = np.arange(num_modes + 1)

    fig, axes = plt.subplots(2, 4, figsize=(14, 8))

    for i, ax in enumerate(axes.flat):
        shape = np.insert(mode_shapes[:, i], 0, 0)  # Add zero at base
        ax.plot(shape, floors, "b-o", lw=2, ms=6)
        ax.axvline(0, color="k", ls="--", lw=0.5)
        ax.set_xlabel("Amplitude")
        ax.set_ylabel("Floor")
        ax.set_title(f"Mode {i + 1} (T={periods[i]:.3f}s)")
        ax.set_ylim(0, num_modes)
        ax.set_yticks(range(num_modes + 1))
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def main():
    # Building parameters
    NUM_STORIES = 8
    MASS = 345.6  # tons
    STIFFNESS = 3.404e5  # kN/m

    # Run analysis
    M, K = build_matrices(NUM_STORIES, MASS, STIFFNESS)
    periods, mode_shapes = modal_analysis(M, K)

    # Print results
    print("Modal Periods:")
    for i, T in enumerate(periods, 1):
        print(f"  Mode {i}: T = {T:.4f} s")

    # Plot and save
    fig = plot_mode_shapes(periods, mode_shapes)
    fig.savefig("mode_shapes.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
