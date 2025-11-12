import matplotlib.pyplot as plt
import numpy as np


class SDOFSystem:
    """SDOF system with optional hysteretic damper"""

    def __init__(self, T=0.4, zeta=0.05, m=1.0):
        self.m = m
        self.omega_n = 2 * np.pi / T
        self.k = m * self.omega_n**2
        self.c = 2 * zeta * m * self.omega_n

        # Hysteretic damper
        self.has_damper = False
        self.k_bar = 0
        self.F_y = 0
        self.damper_force = 0
        self.damper_disp = 0

    def add_damper(self, k_bar_ratio, F_y):
        self.has_damper = True
        self.k_bar = k_bar_ratio * self.k
        self.F_y = F_y
        self.damper_force = 0
        self.damper_disp = 0

    def update_damper(self, disp):
        if not self.has_damper:
            return 0

        delta_disp = disp - self.damper_disp
        trial_force = self.damper_force + self.k_bar * delta_disp

        if abs(trial_force) > self.F_y:
            self.damper_force = self.F_y * np.sign(trial_force)
        else:
            self.damper_force = trial_force

        self.damper_disp = disp
        return self.damper_force

    def newmark_beta(self, ag, dt, beta=0.25, gamma=0.5):
        n = len(ag)
        u = np.zeros(n)
        v = np.zeros(n)
        a = np.zeros(n)

        # Energy arrays
        E_input = np.zeros(n)
        E_kinetic = np.zeros(n)
        E_elastic = np.zeros(n)
        E_damping = np.zeros(n)
        E_hysteretic = np.zeros(n)

        # Forces for base shear
        base_shear = np.zeros(n)

        # Initial conditions
        a[0] = -ag[0]

        for i in range(n - 1):
            # Predictor
            u_pred = u[i] + dt * v[i] + dt**2 * (0.5 - beta) * a[i]
            v_pred = v[i] + dt * (1 - gamma) * a[i]

            # Damper force
            f_damper = self.update_damper(u_pred) if self.has_damper else 0

            # Corrector
            a[i + 1] = (
                -self.m * ag[i + 1] - self.c * v_pred - self.k * u_pred - f_damper
            ) / self.m
            v[i + 1] = v_pred + dt * gamma * a[i + 1]
            u[i + 1] = u_pred + dt**2 * beta * a[i + 1]

            # Base shear = spring + damping + hysteretic
            base_shear[i + 1] = self.k * u[i + 1] + self.c * v[i + 1] + f_damper

            # Energy calculations
            E_input[i + 1] = E_input[i] - self.m * ag[i + 1] * (u[i + 1] - u[i])
            E_kinetic[i + 1] = 0.5 * self.m * v[i + 1] ** 2
            E_elastic[i + 1] = 0.5 * self.k * u[i + 1] ** 2
            E_damping[i + 1] = E_damping[i] + self.c * v[i + 1] ** 2 * dt

            if self.has_damper and i > 0:
                E_hysteretic[i + 1] = E_hysteretic[i] + f_damper * (u[i + 1] - u[i])

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
    """Load earthquake acceleration data"""
    with open(filename, "r") as f:
        lines = f.readlines()

    # Parse header for NPTS and DT
    npts = None
    dt = None
    for line in lines[:5]:  # Check first 5 lines
        if "NPTS=" in line:
            # Extract NPTS value
            idx = line.find("NPTS=")
            temp = line[idx + 5 :].strip()
            npts_str = temp.split(",")[0].strip()
            npts = int(npts_str)

            # Extract DT value
            idx = line.find("DT=")
            temp = line[idx + 3 :].strip()
            dt_str = temp.split()[0].strip()
            dt = float(dt_str)
            break

    # Read acceleration values
    acc = []
    for line in lines[4:]:  # Start after header
        values = line.strip().split()
        for v in values:
            try:
                acc.append(float(v))
            except:
                continue

    acc = np.array(acc[:npts])
    time = np.arange(len(acc)) * dt

    return time, acc * 9.81  # Convert g to m/s^2


def analyze_system(system, eq_file, eq_name):
    """Analyze system under earthquake"""
    time, ag = load_earthquake(eq_file)
    dt = time[1] - time[0]

    results = system.newmark_beta(ag, dt)

    max_disp = np.max(np.abs(results["displacement"]))
    max_shear = np.max(np.abs(results["base_shear"]))

    print(f"\n{eq_name}:")
    print(f"  Max displacement: {max_disp:.4f} m")
    print(f"  Max base shear: {max_shear:.2f} N")

    return time, results


def plot_response(time, results, title):
    """Create response plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Displacement
    axes[0, 0].plot(time, results["displacement"], "b-", linewidth=1.5)
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Displacement (m)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title("Displacement Response")

    # Base shear
    axes[0, 1].plot(time, results["base_shear"], "r-", linewidth=1.5)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Base Shear (N)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title("Base Shear")

    # Energy components
    axes[1, 0].plot(time, results["E_input"], label="Input", linewidth=2)
    axes[1, 0].plot(time, results["E_kinetic"], label="Kinetic")
    axes[1, 0].plot(time, results["E_elastic"], label="Elastic")
    axes[1, 0].plot(time, results["E_damping"], label="Damping")
    if np.any(results["E_hysteretic"]):
        axes[1, 0].plot(time, results["E_hysteretic"], label="Hysteretic")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Energy (J)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc="best")
    axes[1, 0].set_title("Energy Components")

    # Cumulative energies
    axes[1, 1].plot(time, results["E_input"], "k-", label="Total Input", linewidth=2)
    axes[1, 1].plot(time, results["E_damping"], "b--", label="Damping")
    if np.any(results["E_hysteretic"]):
        axes[1, 1].plot(time, results["E_hysteretic"], "r--", label="Hysteretic")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Cumulative Energy (J)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc="best")
    axes[1, 1].set_title("Cumulative Energy Dissipation")

    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("SDOF SEISMIC ANALYSIS")
    print("=" * 60)

    # Part 1: Bare structure
    print("\nPART 1: SYSTEM WITHOUT DAMPER")
    print("-" * 40)

    system = SDOFSystem(T=0.4, zeta=0.05, m=1.0)
    print(f"System properties:")
    print(f"  Natural period T = 0.4 s")
    print(f"  Damping ratio ζ = 5%")
    print(f"  Stiffness k = {system.k:.2f} N/m")

    # Analyze earthquakes
    time1, results1 = analyze_system(system, "I-ELC180_AT2.txt", "El Centro")
    time2, results2 = analyze_system(system, "DAY-TR_AT2.txt", "Tabas")

    # Find Fbs
    Fbs1 = np.max(np.abs(results1["base_shear"]))
    Fbs2 = np.max(np.abs(results2["base_shear"]))
    Fbs = min(Fbs1, Fbs2)

    print(f"\nBase shear comparison:")
    print(f"  El Centro max: {Fbs1:.2f} N")
    print(f"  Tabas max: {Fbs2:.2f} N")
    print(f"  Minimum (Fbs): {Fbs:.2f} N")

    # Plot bare structure
    plot_response(time1, results1, "Bare Structure - El Centro")
    plot_response(time2, results2, "Bare Structure - Tabas")

    # Part 2: With dampers
    print("\n" + "=" * 60)
    print("PART 2: SYSTEM WITH HYSTERETIC DAMPERS")
    print("=" * 60)

    k_bar_ratios = [0.1, 0.5, 1.0]
    F_y = 0.4 * Fbs
    print(f"Damper yield force F_y = 0.4 × Fbs = {F_y:.2f} N")

    for k_ratio in k_bar_ratios:
        print(f"\n*** Damper stiffness k̄ = {k_ratio}k ***")
        print("-" * 40)

        # El Centro
        system_d1 = SDOFSystem(T=0.4, zeta=0.05, m=1.0)
        system_d1.add_damper(k_ratio, F_y)
        time1_d, results1_d = analyze_system(
            system_d1, "I-ELC180_AT2.txt", f"El Centro (k̄={k_ratio}k)"
        )

        # Tabas
        system_d2 = SDOFSystem(T=0.4, zeta=0.05, m=1.0)
        system_d2.add_damper(k_ratio, F_y)
        time2_d, results2_d = analyze_system(
            system_d2, "DAY-TR_AT2.txt", f"Tabas (k̄={k_ratio}k)"
        )

        # Energy dissipation
        E_hyst_1 = results1_d["E_hysteretic"][-1]
        E_damp_1 = results1_d["E_damping"][-1]
        E_total_1 = results1_d["E_input"][-1]

        if E_total_1 > 0:
            print(f"\nEnergy dissipation (El Centro):")
            print(f"  Hysteretic: {E_hyst_1:.1f} J ({100 * E_hyst_1 / E_total_1:.1f}%)")
            print(f"  Viscous: {E_damp_1:.1f} J ({100 * E_damp_1 / E_total_1:.1f}%)")

        # Plot with damper
        plot_response(time1_d, results1_d, f"With Damper (k̄={k_ratio}k) - El Centro")
        plot_response(time2_d, results2_d, f"With Damper (k̄={k_ratio}k) - Tabas")

    plt.show()


if __name__ == "__main__":
    main()
