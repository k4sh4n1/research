import os  # For directory creation

import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops


class SeismicAnalysis:
    """SDOF system seismic analysis with optional hysteretic device"""

    def __init__(self):
        """Initialize system parameters"""
        self.m = 1.0  # Unit mass
        self.T = 0.4  # Natural period (sec)
        self.zeta = 0.05  # Damping ratio (5%)

        # Derived parameters
        self.omega = 2 * np.pi / self.T
        self.k = self.omega**2 * self.m
        self.c = 2 * self.zeta * self.omega * self.m

        # Storage for results
        self.Fbs = None  # Base shear threshold for device design

        # Create results directory if it doesn't exist
        self.results_dir = "results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def read_seismic_record(self, filename):
        """Read seismic record from file and normalize to 0.4g"""
        with open(filename, "r") as f:
            lines = f.readlines()

        # Parse header for NPTS and DT
        npts, dt = None, None
        for i in range(4):  # Check first 4 header lines
            if "NPTS=" in lines[i]:
                parts = lines[i].split(",")
                for part in parts:
                    if "NPTS=" in part:
                        npts = int(part.split("=")[1])
                    elif "DT=" in part:
                        dt = float(part.split("=")[1].replace("SEC", "").strip())
                break

        # Read acceleration data
        acc = []
        for line in lines[4:]:  # Skip 4 header lines
            values = line.strip().split()
            acc.extend([float(v) for v in values if v])

        acc = np.array(acc[:npts])

        # Normalize to 0.4g
        g = 9.81  # m/s²
        max_acc = np.max(np.abs(acc))
        acc = acc * (0.4 * g / max_acc)

        time = np.arange(len(acc)) * dt

        return time, acc, dt

    def analyze_system_alone(self, time, acc, dt):
        """Analyze SDOF system without device using OpenSeesPy"""
        ops.wipe()
        ops.model("basic", "-ndm", 1, "-ndf", 1)

        # Create nodes
        ops.node(1, 0.0)  # Fixed base
        ops.node(2, 0.0)  # Mass node

        # Boundary conditions
        ops.fix(1, 1)

        # Mass
        ops.mass(2, self.m)

        # Spring element
        ops.uniaxialMaterial("Elastic", 1, self.k)
        ops.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 1)

        alpha_M = 2 * self.zeta * self.omega
        beta_K = 0.0

        # Damping
        ops.rayleigh(alpha_M, beta_K, 0.0, 0.0)

        # Time series for ground motion
        ops.timeSeries("Path", 1, "-dt", dt, "-values", *acc)
        ops.pattern("UniformExcitation", 1, 1, "-accel", 1)

        # Analysis settings
        ops.wipeAnalysis()
        ops.constraints("Plain")
        ops.numberer("Plain")
        ops.system("BandGeneral")
        ops.test("NormDispIncr", 1e-8, 10)
        ops.algorithm("Newton")
        ops.integrator("Newmark", 0.5, 0.25)
        ops.analysis("Transient")

        # Initialize arrays
        n_steps = len(time)
        disp = np.zeros(n_steps)
        vel = np.zeros(n_steps)
        base_shear = np.zeros(n_steps)

        # Run analysis
        for i in range(n_steps):
            ops.analyze(1, dt)
            disp[i] = ops.nodeDisp(2, 1)
            vel[i] = ops.nodeVel(2, 1)
            ops.reactions()
            base_shear[i] = -ops.nodeReaction(1, 1)

        # Calculate energies
        E_kinetic = 0.5 * self.m * vel**2
        E_elastic = 0.5 * self.k * disp**2

        E_damping = np.zeros(n_steps)
        E_input = np.zeros(n_steps)

        for i in range(1, n_steps):
            # Viscous damping energy (cumulative)
            E_damping[i] = (
                E_damping[i - 1] + 0.5 * self.c * (vel[i - 1] ** 2 + vel[i] ** 2) * dt
            )

            # Input energy (cumulative)
            du = disp[i] - disp[i - 1]
            E_input[i] = E_input[i - 1] - 0.5 * self.m * (acc[i - 1] + acc[i]) * du

        return {
            "time": time,
            "disp": disp,
            "base_shear": base_shear,
            "E_kinetic": E_kinetic,
            "E_elastic": E_elastic,
            "E_damping": E_damping,
            "E_input": E_input,
            "max_disp": np.max(np.abs(disp)),
            "max_base_shear": np.max(np.abs(base_shear)),
        }

    def analyze_with_device(self, time, acc, dt, k_bar_ratio, F_y_bar):
        """Analyze SDOF system with hysteretic device"""
        ops.wipe()
        ops.model("basic", "-ndm", 1, "-ndf", 1)

        # Create nodes
        ops.node(1, 0.0)  # Fixed base
        ops.node(2, 0.0)  # Mass node

        # Boundary conditions
        ops.fix(1, 1)

        # Mass
        ops.mass(2, self.m)

        # Main spring element
        ops.uniaxialMaterial("Elastic", 1, self.k)
        ops.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 1)

        # Hysteretic device (elastic-perfectly-plastic)
        k_bar = k_bar_ratio * self.k
        eps_yP = F_y_bar / k_bar  # Yield strain
        ops.uniaxialMaterial("ElasticPP", 2, k_bar, eps_yP)
        ops.element("zeroLength", 2, 1, 2, "-mat", 2, "-dir", 1)

        alpha_M = 2 * self.zeta * self.omega
        beta_K = 0.0

        # Damping
        ops.rayleigh(alpha_M, beta_K, 0.0, 0.0)

        # Time series for ground motion
        ops.timeSeries("Path", 1, "-dt", dt, "-values", *acc)
        ops.pattern("UniformExcitation", 1, 1, "-accel", 1)

        # Analysis settings
        ops.wipeAnalysis()
        ops.constraints("Plain")
        ops.numberer("Plain")
        ops.system("BandGeneral")
        ops.test("NormDispIncr", 1e-8, 10)
        ops.algorithm("Newton")
        ops.integrator("Newmark", 0.5, 0.25)
        ops.analysis("Transient")

        # Initialize arrays
        n_steps = len(time)
        disp = np.zeros(n_steps)
        vel = np.zeros(n_steps)
        base_shear = np.zeros(n_steps)
        device_force = np.zeros(n_steps)

        # Run analysis
        for i in range(n_steps):
            ops.analyze(1, dt)
            disp[i] = ops.nodeDisp(2, 1)
            vel[i] = ops.nodeVel(2, 1)
            ops.reactions()
            base_shear[i] = -ops.nodeReaction(1, 1)
            device_force[i] = ops.eleForce(2, 2)  # Get from second node

        # Calculate energies
        E_kinetic = 0.5 * self.m * vel**2
        E_elastic = 0.5 * self.k * disp**2

        E_damping = np.zeros(n_steps)
        E_hysteresis = np.zeros(n_steps)
        E_input = np.zeros(n_steps)

        # Trapezoidal integration to maintain consistency with the Newmark integrator:
        for i in range(1, n_steps):
            # Viscous damping energy: ∫ c·u̇² dt
            # Trapezoidal rule: (dt/2) * [f(t_i-1) + f(t_i)]
            E_damping[i] = (
                E_damping[i - 1] + 0.5 * self.c * (vel[i - 1] ** 2 + vel[i] ** 2) * dt
            )

            # Hysteretic energy: ∫ fh·u̇ dt = ∫ fh·(du/dt) dt = ∫ fh·du
            # Trapezoidal approximation for fh over interval displacement
            du = disp[i] - disp[i - 1]
            E_hysteresis[i] = (
                E_hysteresis[i - 1] + 0.5 * (device_force[i - 1] + device_force[i]) * du
            )

            # Input energy: -∫ m·üg·u̇ dt = -∫ m·üg·(du/dt) dt = -∫ m·üg·du
            # Trapezoidal approximation for üg over interval displacement
            E_input[i] = E_input[i - 1] - 0.5 * self.m * (acc[i - 1] + acc[i]) * du

        return {
            "time": time,
            "disp": disp,
            "base_shear": base_shear,
            "device_force": device_force,
            "E_kinetic": E_kinetic,
            "E_elastic": E_elastic,
            "E_damping": E_damping,
            "E_hysteresis": E_hysteresis,
            "E_input": E_input,
            "max_disp": np.max(np.abs(disp)),
            "max_base_shear": np.max(np.abs(base_shear)),
        }

    def plot_seismic_records(self, records, filenames):
        """Plot normalized seismic records"""
        fig, ax = plt.subplots(figsize=(10, 5))

        colors = ["#1f77b4", "#ff7f0e"]
        for i, (rec, fname) in enumerate(zip(records, filenames)):
            ax.plot(
                rec["time"],
                rec["acc"],
                label=fname,
                color=colors[i],
                linewidth=1.2,
                alpha=0.8,
            )

        ax.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Acceleration (m/s²)", fontsize=11, fontweight="bold")
        ax.set_title(
            "Normalized Seismic Records (0.4g)", fontsize=12, fontweight="bold", pad=15
        )
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")

        plt.tight_layout()
        filename = os.path.join(self.results_dir, "seismic_records.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_system_alone(self, result, record_name):
        """Plot results for system without device"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Energy components
        ax1.plot(result["time"], result["E_input"], "k-", label="Input", linewidth=1.5)
        ax1.plot(
            result["time"],
            result["E_kinetic"],
            color="#1f77b4",
            label="Kinetic",
            linewidth=1.2,
        )
        ax1.plot(
            result["time"],
            result["E_damping"],
            color="#d62728",
            label="Damping",
            linewidth=1.2,
        )
        ax1.plot(
            result["time"],
            result["E_elastic"],
            color="#2ca02c",
            label="Elastic",
            linewidth=1.2,
        )
        ax1.set_xlabel("Time (s)", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Energy (J)", fontsize=10, fontweight="bold")
        ax1.set_title(
            f"Energy Components - {record_name}", fontsize=11, fontweight="bold"
        )
        ax1.legend(fontsize=9, loc="best", framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle="--")

        # Energy balance
        E_sum = result["E_kinetic"] + result["E_damping"] + result["E_elastic"]
        ax2.plot(result["time"], result["E_input"], "k-", label="Input", linewidth=1.5)
        ax2.plot(result["time"], E_sum, "r--", label="Sum", linewidth=1.5, alpha=0.8)
        ax2.set_xlabel("Time (s)", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Energy (J)", fontsize=10, fontweight="bold")
        ax2.set_title(f"Energy Balance - {record_name}", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=9, loc="best", framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle="--")

        plt.tight_layout()
        filename = os.path.join(
            self.results_dir, f"system_alone_{record_name.replace('.txt', '')}.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_with_device(self, result_alone, result_device, k_bar_ratio, record_name):
        """Plot comparison for system with device"""
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Displacement comparison
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(
            result_alone["time"],
            result_alone["disp"],
            color="#1f77b4",
            label="Without Device",
            linewidth=1.2,
            alpha=0.7,
        )
        ax1.plot(
            result_device["time"],
            result_device["disp"],
            color="#d62728",
            label="With Device",
            linewidth=1.2,
        )

        # Mark max/min for displacement - WITHOUT DEVICE
        idx_max_alone = np.argmax(result_alone["disp"])
        idx_min_alone = np.argmin(result_alone["disp"])
        ax1.plot(
            result_alone["time"][idx_max_alone],
            result_alone["disp"][idx_max_alone],
            "o",
            color="#1f77b4",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=5,
        )
        ax1.plot(
            result_alone["time"][idx_min_alone],
            result_alone["disp"][idx_min_alone],
            "o",
            color="#1f77b4",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=5,
        )
        ax1.text(
            result_alone["time"][idx_max_alone],
            result_alone["disp"][idx_max_alone],
            f"  Max: {result_alone['disp'][idx_max_alone]:.4f}m",
            fontsize=8,
            verticalalignment="bottom",
            color="#1f77b4",
            fontweight="bold",
        )
        ax1.text(
            result_alone["time"][idx_min_alone],
            result_alone["disp"][idx_min_alone],
            f"  Min: {result_alone['disp'][idx_min_alone]:.4f}m",
            fontsize=8,
            verticalalignment="top",
            color="#1f77b4",
            fontweight="bold",
        )

        # Mark max/min for displacement - WITH DEVICE
        idx_max_dev = np.argmax(result_device["disp"])
        idx_min_dev = np.argmin(result_device["disp"])
        ax1.plot(
            result_device["time"][idx_max_dev],
            result_device["disp"][idx_max_dev],
            "s",
            color="#d62728",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=5,
        )
        ax1.plot(
            result_device["time"][idx_min_dev],
            result_device["disp"][idx_min_dev],
            "s",
            color="#d62728",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=5,
        )
        ax1.text(
            result_device["time"][idx_max_dev],
            result_device["disp"][idx_max_dev],
            f"  Max: {result_device['disp'][idx_max_dev]:.4f}m",
            fontsize=8,
            verticalalignment="bottom",
            color="#d62728",
            fontweight="bold",
        )
        ax1.text(
            result_device["time"][idx_min_dev],
            result_device["disp"][idx_min_dev],
            f"  Min: {result_device['disp'][idx_min_dev]:.4f}m",
            fontsize=8,
            verticalalignment="top",
            color="#d62728",
            fontweight="bold",
        )

        ax1.set_xlabel("Time (s)", fontsize=9, fontweight="bold")
        ax1.set_ylabel("Displacement (m)", fontsize=9, fontweight="bold")
        ax1.set_title(
            f"Displacement (k̄/k = {k_bar_ratio})", fontsize=10, fontweight="bold"
        )
        ax1.legend(fontsize=8, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle="--")

        # Base shear comparison
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(
            result_alone["time"],
            result_alone["base_shear"],
            color="#1f77b4",
            label="Without Device",
            linewidth=1.2,
            alpha=0.7,
        )
        ax2.plot(
            result_device["time"],
            result_device["base_shear"],
            color="#d62728",
            label="With Device",
            linewidth=1.2,
        )

        # Mark max/min for base shear - WITHOUT DEVICE
        idx_max_bs_alone = np.argmax(result_alone["base_shear"])
        idx_min_bs_alone = np.argmin(result_alone["base_shear"])
        ax2.plot(
            result_alone["time"][idx_max_bs_alone],
            result_alone["base_shear"][idx_max_bs_alone],
            "o",
            color="#1f77b4",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=5,
        )
        ax2.plot(
            result_alone["time"][idx_min_bs_alone],
            result_alone["base_shear"][idx_min_bs_alone],
            "o",
            color="#1f77b4",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=5,
        )
        ax2.text(
            result_alone["time"][idx_max_bs_alone],
            result_alone["base_shear"][idx_max_bs_alone],
            f"  Max: {result_alone['base_shear'][idx_max_bs_alone]:.2f}N",
            fontsize=8,
            verticalalignment="bottom",
            color="#1f77b4",
            fontweight="bold",
        )
        ax2.text(
            result_alone["time"][idx_min_bs_alone],
            result_alone["base_shear"][idx_min_bs_alone],
            f"  Min: {result_alone['base_shear'][idx_min_bs_alone]:.2f}N",
            fontsize=8,
            verticalalignment="top",
            color="#1f77b4",
            fontweight="bold",
        )

        # Mark max/min for base shear - WITH DEVICE
        idx_max_bs_dev = np.argmax(result_device["base_shear"])
        idx_min_bs_dev = np.argmin(result_device["base_shear"])
        ax2.plot(
            result_device["time"][idx_max_bs_dev],
            result_device["base_shear"][idx_max_bs_dev],
            "s",
            color="#d62728",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=5,
        )
        ax2.plot(
            result_device["time"][idx_min_bs_dev],
            result_device["base_shear"][idx_min_bs_dev],
            "s",
            color="#d62728",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=5,
        )
        ax2.text(
            result_device["time"][idx_max_bs_dev],
            result_device["base_shear"][idx_max_bs_dev],
            f"  Max: {result_device['base_shear'][idx_max_bs_dev]:.2f}N",
            fontsize=8,
            verticalalignment="bottom",
            color="#d62728",
            fontweight="bold",
        )
        ax2.text(
            result_device["time"][idx_min_bs_dev],
            result_device["base_shear"][idx_min_bs_dev],
            f"  Min: {result_device['base_shear'][idx_min_bs_dev]:.2f}N",
            fontsize=8,
            verticalalignment="top",
            color="#d62728",
            fontweight="bold",
        )

        ax2.set_xlabel("Time (s)", fontsize=9, fontweight="bold")
        ax2.set_ylabel("Base Shear (N)", fontsize=9, fontweight="bold")
        ax2.set_title(
            f"Base Shear (k̄/k = {k_bar_ratio})", fontsize=10, fontweight="bold"
        )
        ax2.legend(fontsize=8, framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle="--")

        # Energy components with device
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(
            result_device["time"],
            result_device["E_input"],
            "k-",
            label="Input",
            linewidth=1.3,
        )
        ax3.plot(
            result_device["time"],
            result_device["E_kinetic"],
            color="#1f77b4",
            label="Kinetic",
            linewidth=1,
        )
        ax3.plot(
            result_device["time"],
            result_device["E_damping"],
            color="#d62728",
            label="Damping",
            linewidth=1,
        )
        ax3.plot(
            result_device["time"],
            result_device["E_elastic"],
            color="#2ca02c",
            label="Elastic",
            linewidth=1,
        )
        ax3.plot(
            result_device["time"],
            result_device["E_hysteresis"],
            color="#9467bd",
            label="Hysteresis",
            linewidth=1,
        )
        ax3.set_xlabel("Time (s)", fontsize=9, fontweight="bold")
        ax3.set_ylabel("Energy (J)", fontsize=9, fontweight="bold")
        ax3.set_title("Energy Components", fontsize=10, fontweight="bold")
        ax3.legend(fontsize=8, loc="best", framealpha=0.9, ncol=2)
        ax3.grid(True, alpha=0.3, linestyle="--")

        # Energy balance with device
        ax4 = fig.add_subplot(gs[1, 1])
        E_sum = (
            result_device["E_kinetic"]
            + result_device["E_damping"]
            + result_device["E_elastic"]
            + result_device["E_hysteresis"]
        )
        ax4.plot(
            result_device["time"],
            result_device["E_input"],
            "k-",
            label="Input",
            linewidth=1.3,
        )
        ax4.plot(
            result_device["time"], E_sum, "r--", label="Sum", linewidth=1.3, alpha=0.8
        )
        ax4.set_xlabel("Time (s)", fontsize=9, fontweight="bold")
        ax4.set_ylabel("Energy (J)", fontsize=9, fontweight="bold")
        ax4.set_title("Energy Balance", fontsize=10, fontweight="bold")
        ax4.legend(fontsize=8, framealpha=0.9)
        ax4.grid(True, alpha=0.3, linestyle="--")

        plt.suptitle(f"{record_name}", fontsize=11, fontweight="bold", y=0.995)

        filename = os.path.join(
            self.results_dir,
            f"with_device_{record_name.replace('.txt', '')}_k{k_bar_ratio}.png",
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_hysteresis_loops(self, device_results, record_name):
        """Plot hysteresis loops for all device cases"""
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
        ratios = [0.1, 0.5, 1.0]
        colors = ["#9467bd", "#e377c2", "#8c564b"]

        for i, r in enumerate(ratios):
            res = device_results[r]
            axes[i].plot(
                res["disp"],
                res["device_force"],
                color=colors[i],
                linewidth=1.2,
                alpha=0.9,
            )
            axes[i].set_xlabel("Displacement (m)", fontsize=10, fontweight="bold")
            axes[i].set_ylabel("Device Force (N)", fontsize=10, fontweight="bold")
            axes[i].set_title(
                f"Hysteresis Loop (k̄/k = {r})", fontsize=11, fontweight="bold"
            )
            axes[i].grid(True, alpha=0.3, linestyle="--")
            axes[i].axhline(y=0, color="k", linewidth=0.5, linestyle="-", alpha=0.3)
            axes[i].axvline(x=0, color="k", linewidth=0.5, linestyle="-", alpha=0.3)

        plt.suptitle(
            f"Hysteresis Loops - {record_name}", fontsize=12, fontweight="bold"
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        filename = os.path.join(
            self.results_dir, f"hysteresis_loops_{record_name.replace('.txt', '')}.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    sa = SeismicAnalysis()

    # Seismic record filenames
    filenames = ["record-TABAS", "record-ELCENTRO"]

    # Load records
    t1, acc1, dt1 = sa.read_seismic_record(filenames[0])
    t2, acc2, dt2 = sa.read_seismic_record(filenames[1])

    # Plot seismic records
    sa.plot_seismic_records(
        [{"time": t1, "acc": acc1}, {"time": t2, "acc": acc2}], filenames
    )

    # Analyze system alone for both records
    res1 = sa.analyze_system_alone(t1, acc1, dt1)
    res2 = sa.analyze_system_alone(t2, acc2, dt2)
    sa.plot_system_alone(res1, filenames[0])
    sa.plot_system_alone(res2, filenames[1])

    # Determine F̄_y = 0.4 * F_bs (lowest max base shear)
    sa.Fbs = min(res1["max_base_shear"], res2["max_base_shear"])
    F_y_bar = 0.4 * sa.Fbs

    # Device stiffness ratios
    k_ratios = [0.1, 0.5, 1.0]

    # Helper functions
    def print_results_header(filename, result):
        """Print header and baseline results"""
        print(f"\n{'=' * 60}")
        print(f"Results for {filename}")
        print(f"{'=' * 60}")
        print(f"Without device - Max Displacement: {result['max_disp']:.6f} m")
        print(f"Without device - Max Base Shear: {result['max_base_shear']:.2f} N")
        print(f"{'-' * 60}")

    def print_device_results(result_baseline, result_device, k_ratio):
        """Print results for a single device configuration"""
        disp_change = (
            (result_device["max_disp"] - result_baseline["max_disp"])
            / result_baseline["max_disp"]
        ) * 100
        bs_change = (
            (result_device["max_base_shear"] - result_baseline["max_base_shear"])
            / result_baseline["max_base_shear"]
        ) * 100

        print(f"\nWith device (k̄/k = {k_ratio}):")
        print(
            f"  Max Displacement: {result_device['max_disp']:.6f} m ({disp_change:+.2f}%)"
        )
        print(
            f"  Max Base Shear: {result_device['max_base_shear']:.2f} N ({bs_change:+.2f}%)"
        )

    def analyze_with_devices(
        sa, time, acc, dt, result_baseline, k_ratios, F_y_bar, filename
    ):
        """Analyze system with all device configurations"""
        device_results = {}
        for r in k_ratios:
            res_dev = sa.analyze_with_device(time, acc, dt, r, F_y_bar)
            device_results[r] = res_dev
            print_device_results(result_baseline, res_dev, r)
            sa.plot_with_device(result_baseline, res_dev, r, filename)
        sa.plot_hysteresis_loops(device_results, filename)
        return device_results

    # Analysis for record 1
    print_results_header(filenames[0], res1)
    device_results_1 = analyze_with_devices(
        sa, t1, acc1, dt1, res1, k_ratios, F_y_bar, filenames[0]
    )

    # Analysis for record 2
    print_results_header(filenames[1], res2)
    device_results_2 = analyze_with_devices(
        sa, t2, acc2, dt2, res2, k_ratios, F_y_bar, filenames[1]
    )
