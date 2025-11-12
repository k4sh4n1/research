import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid


class SDOFSystem:
    """SDOF system with optional hysteretic damper"""

    def __init__(self, T=0.4, zeta=0.05, m=1.0):
        """Initialize SDOF system parameters"""
        self.m = m
        self.omega_n = 2 * np.pi / T
        self.k = m * self.omega_n**2
        self.c = 2 * zeta * m * self.omega_n

        # Hysteretic damper parameters
        self.has_damper = False
        self.k_bar = 0
        self.F_y = 0
        self.damper_force = 0
        self.damper_disp = 0
        self.yielded = False

    def add_damper(self, k_bar_ratio, F_y):
        """Add hysteretic damper to system"""
        self.has_damper = True
        self.k_bar = k_bar_ratio * self.k
        self.F_y = F_y
        self.damper_force = 0
        self.damper_disp = 0
        self.yielded = False

    def update_damper(self, disp, dt):
        """Update damper force using bilinear hysteresis model"""
        if not self.has_damper:
            return 0

        delta_disp = disp - self.damper_disp
        trial_force = self.damper_force + self.k_bar * delta_disp

        # Check yielding
        if abs(trial_force) > self.F_y:
            self.damper_force = self.F_y * np.sign(trial_force)
            self.yielded = True
        else:
            self.damper_force = trial_force

        self.damper_disp = disp
        return self.damper_force

    def newmark_beta(self, ag, dt, beta=0.25, gamma=0.5):
        """Newmark-beta time integration"""
        n = len(ag)
        u = np.zeros(n)
        v = np.zeros(n)
        a = np.zeros(n)

        # Energy components
        E_input = np.zeros(n)
        E_kinetic = np.zeros(n)
        E_elastic = np.zeros(n)
        E_damping = np.zeros(n)
        E_hysteretic = np.zeros(n)

        # Initial acceleration
        a[0] = -ag[0]

        for i in range(n - 1):
            # Predictor
            u_pred = u[i] + dt * v[i] + dt**2 * (0.5 - beta) * a[i]
            v_pred = v[i] + dt * (1 - gamma) * a[i]

            # Update damper force if present
            if self.has_damper:
                f_damper = self.update_damper(u_pred, dt)
            else:
                f_damper = 0

            # Corrector
            a[i + 1] = (
                -self.m * ag[i + 1] - self.c * v_pred - self.k * u_pred - f_damper
            ) / self.m
            v[i + 1] = v_pred + dt * gamma * a[i + 1]
            u[i + 1] = u_pred + dt**2 * beta * a[i + 1]

            # Energy calculations
            dE_input = -self.m * ag[i + 1] * (v[i + 1] + v[i]) * dt / 2
            E_input[i + 1] = E_input[i] + dE_input

            E_kinetic[i + 1] = 0.5 * self.m * v[i + 1] ** 2
            E_elastic[i + 1] = 0.5 * self.k * u[i + 1] ** 2

            dE_damping = self.c * v[i + 1] ** 2 * dt
            E_damping[i + 1] = E_damping[i] + dE_damping

            if self.has_damper:
                dE_hysteretic = f_damper * (u[i + 1] - u[i])
                E_hysteretic[i + 1] = E_hysteretic[i] + dE_hysteretic

        # Calculate base shear
        base_shear = self.k * u + self.c * v
        if self.has_damper:
            base_shear += np.array([self.damper_force for _ in range(n)])

        return {
            "displacement": u,
            "velocity": v,
            "acceleration": a,
            "base_shear": base_shear,
            "E_input": E_input,
            "E_kinetic": E_kinetic,
            "E_elastic": E_elastic,
            "E_damping": E_damping,
            "E_hysteretic": E_hysteretic,
        }


def load_earthquake(filename):
    """Load earthquake acceleration time history"""
    with open(filename, "r") as f:
        lines = f.readlines()

    # Find NPTS and DT
    for line in lines:
        if "NPTS=" in line:
            parts = line.split()
            for part in parts:
                if "NPTS=" in part:
                    npts = int(part.split("=")[1].replace(",", ""))
                elif "DT=" in part:
                    dt = float(part.split("=")[1])
            break

    # Extract acceleration data
    acc = []
    for line in lines[4:]:  # Skip header lines
        values = line.split()
        acc.extend([float(v) for v in values])

    acc = np.array(acc[:npts])
    time = np.arange(len(acc)) * dt

    return time, acc * 9.81  # Convert from g to m/s^2


def analyze_system(system, eq_file, eq_name):
    """Analyze system response to earthquake"""
    time, ag = load_earthquake(eq_file)
    dt = time[1] - time[0]

    results = system.newmark_beta(ag, dt)

    print(f"\n{eq_name}:")
    print(f"  Max displacement: {np.max(np.abs(results['displacement'])):.4f} m")
    print(f"  Max base shear: {np.max(np.abs(results['base_shear'])):.2f} N")

    return time, results


def plot_response(time, results, title):
    """Plot system response"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)

    # Displacement
    axes[0, 0].plot(time, results["displacement"])
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Displacement (m)")
    axes[0, 0].grid(True)
    axes[0, 0].set_title("Displacement Response")

    # Base shear
    axes[0, 1].plot(time, results["base_shear"])
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Base Shear (N)")
    axes[0, 1].grid(True)
    axes[0, 1].set_title("Base Shear")

    # Energy components
    axes[1, 0].plot(time, results["E_input"], label="Input")
    axes[1, 0].plot(time, results["E_kinetic"], label="Kinetic")
    axes[1, 0].plot(time, results["E_elastic"], label="Elastic")
    axes[1, 0].plot(time, results["E_damping"], label="Damping")
    if np.any(results["E_hysteretic"]):
        axes[1, 0].plot(time, results["E_hysteretic"], label="Hysteretic")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Energy (J)")
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    axes[1, 0].set_title("Energy Components")

    # Energy balance check
    E_total = (
        results["E_kinetic"]
        + results["E_elastic"]
        + results["E_damping"]
        + results["E_hysteretic"]
    )
    axes[1, 1].plot(time, results["E_input"], label="Input", linewidth=2)
    axes[1, 1].plot(time, E_total, label="Sum of components", linestyle="--")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Energy (J)")
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    axes[1, 1].set_title("Energy Balance Check")

    plt.tight_layout()
    return fig


# Main analysis
def main():
    print("=" * 60)
    print("SDOF SEISMIC ANALYSIS")
    print("=" * 60)

    # Part 1: System without damper
    print("\nPART 1: SYSTEM WITHOUT DAMPER")
    print("-" * 40)

    system = SDOFSystem(T=0.4, zeta=0.05, m=1.0)

    # Analyze both earthquakes
    time1, results1 = analyze_system(system, "I-ELC180_AT2.txt", "El Centro")
    time2, results2 = analyze_system(system, "DAY-TR_AT2.txt", "Tabas")

    # Find minimum max base shear
    Fbs1 = np.max(np.abs(results1["base_shear"]))
    Fbs2 = np.max(np.abs(results2["base_shear"]))
    Fbs = min(Fbs1, Fbs2)

    print(f"\nMinimum max base shear (Fbs): {Fbs:.2f} N")

    # Plot results for system without damper
    fig1 = plot_response(time1, results1, "System without Damper - El Centro")
    fig2 = plot_response(time2, results2, "System without Damper - Tabas")

    # Part 2: System with dampers
    print("\n" + "=" * 60)
    print("PART 2: SYSTEM WITH HYSTERETIC DAMPERS")
    print("=" * 60)

    k_bar_ratios = [0.1, 0.5, 1.0]
    F_y = 0.4 * Fbs

    for k_ratio in k_bar_ratios:
        print(f"\n*** k_bar = {k_ratio}k, F_y = {F_y:.2f} N ***")
        print("-" * 40)

        # El Centro with damper
        system_d1 = SDOFSystem(T=0.4, zeta=0.05, m=1.0)
        system_d1.add_damper(k_ratio, F_y)
        time1_d, results1_d = analyze_system(
            system_d1, "I-ELC180_AT2.txt", "El Centro with Damper"
        )

        # Tabas with damper
        system_d2 = SDOFSystem(T=0.4, zeta=0.05, m=1.0)
        system_d2.add_damper(k_ratio, F_y)
        time2_d, results2_d = analyze_system(
            system_d2, "DAY-TR_AT2.txt", "Tabas with Damper"
        )

        # Plot results
        fig3 = plot_response(
            time1_d, results1_d, f"El Centro with Damper (k_bar={k_ratio}k)"
        )
        fig4 = plot_response(
            time2_d, results2_d, f"Tabas with Damper (k_bar={k_ratio}k)"
        )

    plt.show()

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Natural period T = 0.4 sec")
    print(f"Damping ratio ζ = 5%")
    print(f"System stiffness k = {system.k:.2f} N/m")
    print(f"Yield force F_y = {F_y:.2f} N")


if __name__ == "__main__":
    main()
