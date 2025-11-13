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

        # Damping
        ops.rayleigh(0.0, 0.0, 0.0, 2 * self.zeta / self.omega)

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
        ops.uniaxialMaterial("Steel01", 2, F_y_bar, k_bar, 0.001)
        ops.element("zeroLength", 2, 1, 2, "-mat", 2, "-dir", 1)

        # Damping
        ops.rayleigh(0.0, 0.0, 0.0, 2 * self.zeta / self.omega)

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
            device_force[i] = ops.eleForce(2, 1)

        # Calculate energies
        E_kinetic = 0.5 * self.m * vel**2
        E_elastic = 0.5 * self.k * disp**2

        E_damping = np.zeros(n_steps)
        E_hysteresis = np.zeros(n_steps)
        E_input = np.zeros(n_steps)

        for i in range(1, n_steps):
            # Viscous damping energy
            E_damping[i] = (
                E_damping[i - 1] + 0.5 * self.c * (vel[i - 1] ** 2 + vel[i] ** 2) * dt
            )

            # Hysteretic energy
            du = disp[i] - disp[i - 1]
            E_hysteresis[i] = (
                E_hysteresis[i - 1] + 0.5 * (device_force[i - 1] + device_force[i]) * du
            )

            # Input energy
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

    def plot_seismic_records(self, records):
        """Plot normalized seismic records"""
        plt.figure(figsize=(10, 4))
        for i, rec in enumerate(records, 1):
            plt.plot(rec["time"], rec["acc"], label=f"Seismic Record {i}", alpha=0.7)
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (m/s²)")
        plt.title("Normalized Seismic Records (0.4g)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_system_alone(self, result):
        """Plot results for system without device"""
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Energy components
        axes[0].plot(result["time"], result["E_input"], "k-", label="Input Energy")
        axes[0].plot(result["time"], result["E_kinetic"], "b-", label="Kinetic Energy")
        axes[0].plot(result["time"], result["E_damping"], "r-", label="Damping Energy")
        axes[0].plot(result["time"], result["E_elastic"], "g-", label="Elastic Energy")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Energy (J)")
        axes[0].set_title("Energy Components")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Energy balance
        E_sum = result["E_kinetic"] + result["E_damping"] + result["E_elastic"]
        axes[1].plot(result["time"], result["E_input"], "k-", label="Input Energy")
        axes[1].plot(result["time"], E_sum, "r--", label="Sum of Components")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Energy (J)")
        axes[1].set_title("Energy Balance Check")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_with_device(self, result_alone, result_device, k_bar_ratio):
        """Plot comparison for system with device"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Displacement comparison
        axes[0, 0].plot(
            result_alone["time"], result_alone["disp"], "b-", label="Without Device"
        )
        axes[0, 0].plot(
            result_device["time"], result_device["disp"], "r-", label="With Device"
        )
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Displacement (m)")
        axes[0, 0].set_title(f"Displacement Comparison (k̄/k = {k_bar_ratio})")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Base shear comparison
        axes[0, 1].plot(
            result_alone["time"],
            result_alone["base_shear"],
            "b-",
            label="Without Device",
        )
        axes[0, 1].plot(
            result_device["time"],
            result_device["base_shear"],
            "r-",
            label="With Device",
        )
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Base Shear (N)")
        axes[0, 1].set_title(f"Base Shear Comparison (k̄/k = {k_bar_ratio})")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Energy components with device
        axes[1, 0].plot(
            result_device["time"], result_device["E_input"], "k-", label="Input"
        )
        axes[1, 0].plot(
            result_device["time"], result_device["E_kinetic"], "b-", label="Kinetic"
        )
        axes[1, 0].plot(
            result_device["time"], result_device["E_damping"], "r-", label="Damping"
        )
        axes[1, 0].plot(
            result_device["time"], result_device["E_elastic"], "g-", label="Elastic"
        )
        axes[1, 0].plot(
            result_device["time"],
            result_device["E_hysteresis"],
            "m-",
            label="Hysteresis",
        )
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("Energy (J)")
        axes[1, 0].set_title("Energy Components")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Energy balance with device
        E_sum = (
            result_device["E_kinetic"]
            + result_device["E_damping"]
            + result_device["E_elastic"]
            + result_device["E_hysteresis"]
        )
        axes[1, 1].plot(
            result_device["time"], result_device["E_input"], "k-", label="Input Energy"
        )
        axes[1, 1].plot(result_device["time"], E_sum, "r--", label="Sum of Components")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Energy (J)")
        axes[1, 1].set_title("Energy Balance Check")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_hysteresis_loops(self, device_results):
        """Plot hysteresis loops for all device cases"""
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
        ratios = [0.1, 0.5, 1.0]
        for i, r in enumerate(ratios):
            res = device_results[r]
            axes[i].plot(res["disp"], res["device_force"], "m-")
            axes[i].set_xlabel("Displacement (m)")
            axes[i].set_ylabel("Device Force (N)")
            axes[i].set_title(f"Hysteresis Loop (k̄/k = {r})")
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# === Execution Example ===
if __name__ == "__main__":
    sa = SeismicAnalysis()

    # Load records
    t1, acc1, dt1 = sa.read_seismic_record("seismic1.txt")
    t2, acc2, dt2 = sa.read_seismic_record("seismic2.txt")

    # Plot seismic records
    sa.plot_seismic_records([{"time": t1, "acc": acc1}, {"time": t2, "acc": acc2}])

    # Analyze system alone for both records
    res1 = sa.analyze_system_alone(t1, acc1, dt1)
    res2 = sa.analyze_system_alone(t2, acc2, dt2)
    sa.plot_system_alone(res1)

    # Determine F̄_y = 0.4 * F_bs (lowest max base shear)
    sa.Fbs = min(res1["max_base_shear"], res2["max_base_shear"])
    F_y_bar = 0.4 * sa.Fbs

    # Device stiffness ratios
    k_ratios = [0.1, 0.5, 1.0]
    device_results = {}

    for r in k_ratios:
        res_dev = sa.analyze_with_device(t1, acc1, dt1, r, F_y_bar)
        device_results[r] = res_dev
        sa.plot_with_device(res1, res_dev, r)

    # Plot hysteresis loops
    sa.plot_hysteresis_loops(device_results)
