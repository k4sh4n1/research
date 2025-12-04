import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops


def create_shear_building(num_stories, mass, stiffness, height):
    """Create a 1D shear building model in OpenSees."""
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)

    # Create nodes (only x-coordinate in 1D, representing floor position)
    for i in range(num_stories + 1):
        ops.node(i, i * height)  # Node tag, x-coordinate

    # Boundary conditions: fix base (node 0)
    ops.fix(0, 1)

    # Assign mass to floor nodes (only 1 DOF per node)
    for i in range(1, num_stories + 1):
        ops.mass(i, mass)

    # Create uniaxial elastic material for story stiffness
    ops.uniaxialMaterial("Elastic", 1, stiffness)

    # Create zeroLength spring elements connecting floors
    # In 1D, we use zeroLength elements with spring stiffness
    for i in range(num_stories):
        ele_tag = i + 1
        node_i = i
        node_j = i + 1
        mat_tag = 1  # Uniaxial elastic material
        # zeroLength element connecting node_i to node_j
        # "-dir 1" means the material acts in DOF 1 (horizontal)
        ops.element("zeroLength", ele_tag, node_i, node_j, "-mat", mat_tag, "-dir", 1)


def run_modal_analysis(num_modes):
    """Run eigenvalue analysis and return periods and mode shapes."""
    eigenvalues = ops.eigen("-fullGenLapack", num_modes)
    periods = 2 * np.pi / np.sqrt(eigenvalues)

    # Extract mode shapes (1 DOF per node now)
    mode_shapes = np.array(
        [
            [
                ops.nodeEigenvector(floor, mode + 1, 1)
                for floor in range(1, num_modes + 1)
            ]
            for mode in range(num_modes)
        ]
    ).T

    # Normalize mode shapes (roof = 1)
    mode_shapes /= mode_shapes[-1, :]

    return periods, mode_shapes


def plot_mode_shapes(
    periods,
    mode_shapes,
    title="Mode Shapes of 8-Story Shear Building (OpenSees)",
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
    HEIGHT = 3.0  # m (used only for node coordinate, not stiffness calculation)

    # Run analysis
    create_shear_building(NUM_STORIES, MASS, STIFFNESS, HEIGHT)
    periods, mode_shapes = run_modal_analysis(NUM_STORIES)

    # Print results
    print("Modal Periods:")
    for i, T in enumerate(periods, 1):
        print(f"  Mode {i}: T = {T:.4f} s")

    # Plot and save
    fig = plot_mode_shapes(periods, mode_shapes)
    fig.savefig("mode_shapes--opensees.png", dpi=150)
    plt.show()

    ops.wipe()


if __name__ == "__main__":
    main()
