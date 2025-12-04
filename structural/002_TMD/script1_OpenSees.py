import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops


def create_shear_building(num_stories, mass, stiffness, height):
    """Create a shear building model in OpenSees."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    # Create nodes
    for i in range(num_stories + 1):
        ops.node(i, 0.0, i * height)

    # Boundary conditions: fix base, allow only horizontal DOF for floors
    ops.fix(0, 1, 1, 1)
    for i in range(1, num_stories + 1):
        ops.fix(i, 0, 1, 1)
        ops.mass(i, mass, 1e-10, 1e-10)

    # Create elements (shear beam: k = 12EI/h³)
    ops.geomTransf("Linear", 1)
    E = 2.1e8
    I = stiffness * height**3 / (12 * E)

    for i in range(num_stories):
        ops.element("elasticBeamColumn", i + 1, i, i + 1, 1.0, E, I, 1)


def run_modal_analysis(num_modes):
    """Run eigenvalue analysis and return periods and mode shapes."""
    eigenvalues = ops.eigen("-fullGenLapack", num_modes)
    periods = 2 * np.pi / np.sqrt(eigenvalues)

    # Extract and normalize mode shapes (roof = 1)
    mode_shapes = np.array(
        [
            [
                ops.nodeEigenvector(floor, mode + 1, 1)
                for floor in range(1, num_modes + 1)
            ]
            for mode in range(num_modes)
        ]
    ).T
    mode_shapes /= mode_shapes[-1, :]

    return periods, mode_shapes


def plot_mode_shapes(
    periods, mode_shapes, title="Mode Shapes of 8-Story Shear Building"
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
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def main():
    # Building parameters
    NUM_STORIES = 8
    MASS = 345.6  # tons
    STIFFNESS = 3.404e5  # kN/m
    HEIGHT = 3.0  # m

    # Run analysis
    create_shear_building(NUM_STORIES, MASS, STIFFNESS, HEIGHT)
    periods, mode_shapes = run_modal_analysis(NUM_STORIES)

    # Print results
    print("Modal Periods:")
    for i, T in enumerate(periods, 1):
        print(f"  Mode {i}: T = {T:.4f} s")

    # Plot and save
    fig = plot_mode_shapes(periods, mode_shapes)
    fig.savefig("mode_shapes_opensees.png", dpi=150)
    plt.show()

    ops.wipe()


if __name__ == "__main__":
    main()
