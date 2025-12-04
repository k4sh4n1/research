import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh


class ShearBuildingTMD:
    """
    8-Story Shear Building with Tuned Mass Damper (TMD) Analysis
    Structural Control - Homework 3
    """

    def __init__(self):
        """Initialize the 8-story shear building parameters"""

        # Number of floors
        self.n_floors = 8

        # Floor properties (same for all floors)
        # Converting to SI units
        self.m_floor = 345.6 * 1000  # kg (345.6 tons)
        self.k_story = 3.404e5 * 1000  # N/m (3.404×10^5 kN/m)
        self.c_story = 2937 * 1000  # N·s/m (2937 tons/s)

        # Create arrays for each floor/story
        self.m = np.ones(self.n_floors) * self.m_floor  # Mass of each floor
        self.k = np.ones(self.n_floors) * self.k_story  # Stiffness of each story
        self.c = np.ones(self.n_floors) * self.c_story  # Damping of each story

        # Build system matrices
        self.M = self._build_mass_matrix()
        self.K = self._build_stiffness_matrix()
        self.C = self._build_damping_matrix()

        # Modal properties (to be computed)
        self.omega = None  # Natural frequencies (rad/s)
        self.freq = None  # Natural frequencies (Hz)
        self.period = None  # Natural periods (s)
        self.phi = None  # Mode shapes
        self.M_modal = None  # Modal masses

        # Create results directory
        self.results_dir = "results_hw3"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def _build_mass_matrix(self):
        """
        Build the mass matrix [M] for the shear building

        Returns:
            M: (n_floors × n_floors) diagonal mass matrix
        """
        M = np.diag(self.m)
        return M

    def _build_stiffness_matrix(self):
        """
        Build the stiffness matrix [K] for the shear building

        For shear building with n floors:
        K[i,i] = k[i] + k[i+1]  (for i < n-1)
        K[n-1,n-1] = k[n-1]    (top floor)
        K[i,i+1] = K[i+1,i] = -k[i+1]

        Returns:
            K: (n_floors × n_floors) tridiagonal stiffness matrix
        """
        n = self.n_floors
        K = np.zeros((n, n))

        for i in range(n):
            # Diagonal terms
            if i == 0:
                # First floor: connected to ground and floor above
                K[i, i] = self.k[0] + self.k[1]
            elif i == n - 1:
                # Top floor: connected only to floor below
                K[i, i] = self.k[i]
            else:
                # Middle floors: connected to floors above and below
                K[i, i] = self.k[i] + self.k[i + 1]

            # Off-diagonal terms
            if i < n - 1:
                K[i, i + 1] = -self.k[i + 1]
                K[i + 1, i] = -self.k[i + 1]

        return K

    def _build_damping_matrix(self):
        """
        Build the damping matrix [C] for the shear building

        Same structure as stiffness matrix but with damping coefficients

        Returns:
            C: (n_floors × n_floors) tridiagonal damping matrix
        """
        n = self.n_floors
        C = np.zeros((n, n))

        for i in range(n):
            # Diagonal terms
            if i == 0:
                C[i, i] = self.c[0] + self.c[1]
            elif i == n - 1:
                C[i, i] = self.c[i]
            else:
                C[i, i] = self.c[i] + self.c[i + 1]

            # Off-diagonal terms
            if i < n - 1:
                C[i, i + 1] = -self.c[i + 1]
                C[i + 1, i] = -self.c[i + 1]

        return C

    def modal_analysis(self):
        """
        Perform modal analysis to find natural frequencies and mode shapes

        Solves the generalized eigenvalue problem:
        [K]{φ} = ω²[M]{φ}

        Returns:
            Dictionary containing:
            - omega: natural frequencies (rad/s)
            - freq: natural frequencies (Hz)
            - period: natural periods (s)
            - phi: mode shapes (normalized to unit modal mass)
            - M_modal: modal masses
        """
        # Solve generalized eigenvalue problem
        # eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = eigh(self.K, self.M)

        # Natural frequencies
        self.omega = np.sqrt(eigenvalues)  # rad/s
        self.freq = self.omega / (2 * np.pi)  # Hz
        self.period = 1 / self.freq  # seconds

        # Mode shapes (eigenvectors)
        # Each column is a mode shape
        self.phi = eigenvectors

        # Normalize mode shapes to roof (floor 8) = 1.0
        # This is specified in the assignment for TMD design
        for i in range(self.n_floors):
            self.phi[:, i] = self.phi[:, i] / self.phi[-1, i]

        # Calculate modal masses: M_i = {φ_i}^T [M] {φ_i}
        self.M_modal = np.zeros(self.n_floors)
        for i in range(self.n_floors):
            self.M_modal[i] = self.phi[:, i].T @ self.M @ self.phi[:, i]

        return {
            "omega": self.omega,
            "freq": self.freq,
            "period": self.period,
            "phi": self.phi,
            "M_modal": self.M_modal,
        }

    def print_modal_results(self):
        """Print the modal analysis results in a formatted table"""

        if self.omega is None:
            self.modal_analysis()

        print("=" * 70)
        print("MODAL ANALYSIS RESULTS - 8-STORY SHEAR BUILDING")
        print("=" * 70)

        # Print system parameters
        print("\nSystem Parameters:")
        print(
            f"  Floor mass (m):      {self.m_floor / 1000:.1f} tons = {self.m_floor:.0f} kg"
        )
        print(
            f"  Story stiffness (k): {self.k_story / 1e6:.3f} ×10⁶ N/m = {self.k_story / 1000:.0f} kN/m"
        )
        print(f"  Story damping (c):   {self.c_story / 1000:.0f} kN·s/m")

        # Print natural frequencies and periods
        print("\n" + "-" * 70)
        print(
            f"{'Mode':<6} {'ω (rad/s)':<14} {'f (Hz)':<12} {'T (s)':<12} {'M_modal (kg)':<15}"
        )
        print("-" * 70)

        for i in range(self.n_floors):
            print(
                f"{i + 1:<6} {self.omega[i]:<14.4f} {self.freq[i]:<12.4f} "
                f"{self.period[i]:<12.4f} {self.M_modal[i]:<15.2f}"
            )

        print("-" * 70)

        # Print mode shapes
        print("\nMode Shapes (normalized to roof = 1.0):")
        print("-" * 70)

        # Header
        header = f"{'Floor':<8}"
        for i in range(self.n_floors):
            header += f"{'Mode ' + str(i + 1):<10}"
        print(header)
        print("-" * 70)

        # Mode shape values
        for floor in range(self.n_floors):
            row = f"{floor + 1:<8}"
            for mode in range(self.n_floors):
                row += f"{self.phi[floor, mode]:<10.4f}"
            print(row)

        print("=" * 70)

    def plot_mode_shapes(self, modes_to_plot=None, save_fig=True):
        """
        Plot the mode shapes of the building

        Args:
            modes_to_plot: list of mode numbers to plot (1-indexed),
                          default is first 4 modes
            save_fig: whether to save the figure
        """
        if self.phi is None:
            self.modal_analysis()

        if modes_to_plot is None:
            modes_to_plot = [1, 2, 3, 4]  # Default: first 4 modes

        # Floor heights (for plotting)
        floors = np.arange(0, self.n_floors + 1)  # 0 = ground, 1-8 = floors

        fig, axes = plt.subplots(
            1, len(modes_to_plot), figsize=(3.5 * len(modes_to_plot), 8)
        )

        if len(modes_to_plot) == 1:
            axes = [axes]

        for idx, mode_num in enumerate(modes_to_plot):
            ax = axes[idx]
            mode_idx = mode_num - 1  # Convert to 0-indexed

            # Mode shape with ground = 0
            mode_shape = np.concatenate([[0], self.phi[:, mode_idx]])

            # Plot mode shape
            ax.plot(mode_shape, floors, "b-o", linewidth=2, markersize=8)
            ax.axvline(x=0, color="k", linestyle="--", linewidth=0.5)

            # Plot floor lines
            for floor in floors:
                ax.axhline(
                    y=floor, color="gray", linestyle="-", linewidth=0.3, alpha=0.5
                )

            ax.set_xlabel("Modal Amplitude", fontsize=11)
            ax.set_ylabel("Floor", fontsize=11)
            ax.set_title(
                f"Mode {mode_num}\n"
                f"T = {self.period[mode_idx]:.3f} s\n"
                f"f = {self.freq[mode_idx]:.3f} Hz",
                fontsize=11,
            )
            ax.set_yticks(floors)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.2, self.n_floors + 0.5)

        plt.suptitle(
            "Mode Shapes - 8-Story Shear Building",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        if save_fig:
            plt.savefig(
                f"{self.results_dir}/mode_shapes.png", dpi=300, bbox_inches="tight"
            )
            print(f"Mode shapes figure saved to {self.results_dir}/mode_shapes.png")

        plt.show()

    def plot_all_mode_shapes(self, save_fig=True):
        """Plot all 8 mode shapes in a single figure"""

        if self.phi is None:
            self.modal_analysis()

        floors = np.arange(0, self.n_floors + 1)

        fig, axes = plt.subplots(2, 4, figsize=(14, 10))
        axes = axes.flatten()

        for mode_idx in range(self.n_floors):
            ax = axes[mode_idx]

            mode_shape = np.concatenate([[0], self.phi[:, mode_idx]])

            ax.plot(mode_shape, floors, "b-o", linewidth=2, markersize=6)
            ax.axvline(x=0, color="k", linestyle="--", linewidth=0.5)

            for floor in floors:
                ax.axhline(
                    y=floor, color="gray", linestyle="-", linewidth=0.3, alpha=0.5
                )

            ax.set_xlabel("Amplitude", fontsize=10)
            ax.set_ylabel("Floor", fontsize=10)
            ax.set_title(
                f"Mode {mode_idx + 1}: T={self.period[mode_idx]:.3f}s, f={self.freq[mode_idx]:.2f}Hz",
                fontsize=10,
            )
            ax.set_yticks(floors)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.2, self.n_floors + 0.5)

        plt.suptitle(
            "All Mode Shapes - 8-Story Shear Building", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        if save_fig:
            plt.savefig(
                f"{self.results_dir}/all_mode_shapes.png", dpi=300, bbox_inches="tight"
            )
            print(
                f"All mode shapes figure saved to {self.results_dir}/all_mode_shapes.png"
            )

        plt.show()


# =============================================================================
# Main execution
# =============================================================================
if __name__ == "__main__":
    # Create the building model
    building = ShearBuildingTMD()

    # Perform modal analysis
    modal_results = building.modal_analysis()

    # Print results
    building.print_modal_results()

    # Plot mode shapes
    building.plot_mode_shapes(modes_to_plot=[1, 2, 3, 4])

    # Plot all mode shapes
    building.plot_all_mode_shapes()
