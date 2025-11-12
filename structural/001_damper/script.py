from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


@dataclass
class SystemParams:
    """SDOF system parameters"""

    m: float = 1.0  # Unit mass
    T: float = 0.4  # Natural period (sec)
    zeta: float = 0.05  # Damping ratio (5%)

    def __post_init__(self):
        self.omega = 2 * np.pi / self.T  # Natural frequency
        self.k = self.omega**2 * self.m  # Stiffness
        self.c = 2 * self.zeta * self.omega * self.m  # Damping coefficient


@dataclass
class DamperParams:
    """Hysteretic damper parameters"""

    k_bar: float  # Damper stiffness
    F_y: float  # Yield force


class SeismicRecord:
    """Load and process seismic acceleration data"""

    def __init__(self, filename: str, skip_lines: int = 4):
        self.acc_raw = self._load_data(filename, skip_lines)
        self.dt = self._extract_dt(filename)
        self.time = np.arange(len(self.acc_raw)) * self.dt

    def _load_data(self, filename: str, skip_lines: int) -> np.ndarray:
        """Load acceleration data from file"""
        with open(filename, "r") as f:
            lines = f.readlines()[skip_lines:]

        acc = []
        for line in lines:
            acc.extend([float(x) for x in line.split()])
        return np.array(acc)

    def _extract_dt(self, filename: str) -> float:
        """Extract time step from file header"""
        with open(filename, "r") as f:
            for line in f:
                if "DT=" in line:
                    return float(line.split("DT=")[1].split()[0])
        return 0.01  # Default

    def normalize(self, target_g: float = 0.4) -> "SeismicRecord":
        """Normalize max acceleration to target value (in g)"""
        max_acc = np.max(np.abs(self.acc_raw))
        self.acc = self.acc_raw * (target_g / max_acc)
        return self


class SDOFAnalysis:
    """SDOF system dynamic analysis"""

    def __init__(self, system: SystemParams):
        self.sys = system

    def solve_elastic(self, record: SeismicRecord) -> Dict:
        """Solve elastic system response"""

        def equations(y, t):
            u, v = y
            ag = np.interp(t, record.time, record.acc)
            a = (
                -(self.sys.omega**2) * u
                - 2 * self.sys.zeta * self.sys.omega * v
                - ag * 9.81
            )
            return [v, a]

        y0 = [0, 0]
        sol = odeint(equations, y0, record.time)

        u = sol[:, 0]
        v = sol[:, 1]

        # Calculate accelerations
        a = np.zeros_like(u)
        for i, t in enumerate(record.time):
            ag = record.acc[i] * 9.81
            a[i] = (
                -(self.sys.omega**2) * u[i]
                - 2 * self.sys.zeta * self.sys.omega * v[i]
                - ag
            )

        # Base shear
        F_base = self.sys.m * (a + record.acc * 9.81)

        # Energy calculations
        E_kinetic = 0.5 * self.sys.m * v**2
        E_elastic = 0.5 * self.sys.k * u**2

        # Damping energy (incremental)
        E_damping = np.zeros_like(u)
        for i in range(1, len(u)):
            E_damping[i] = E_damping[i - 1] + self.sys.c * v[i] ** 2 * record.dt

        # Input energy
        E_input = np.zeros_like(u)
        for i in range(1, len(u)):
            E_input[i] = E_input[i - 1] - self.sys.m * record.acc[i] * 9.81 * (
                u[i] - u[i - 1]
            )

        return {
            "time": record.time,
            "u": u,
            "v": v,
            "a": a,
            "F_base": F_base,
            "u_max": np.max(np.abs(u)),
            "F_max": np.max(np.abs(F_base)),
            "E_kinetic": E_kinetic,
            "E_elastic": E_elastic,
            "E_damping": E_damping,
            "E_input": E_input,
        }

    def solve_with_damper(self, record: SeismicRecord, damper: DamperParams) -> Dict:
        """Solve system with hysteretic damper"""

        def equations(y, t):
            u, v, F_d = y
            ag = np.interp(t, record.time, record.acc)

            # Hysteretic damper force update (bilinear model)
            if abs(F_d) < damper.F_y:
                F_d_new = F_d + damper.k_bar * v * record.dt
                if abs(F_d_new) > damper.F_y:
                    F_d_new = np.sign(F_d_new) * damper.F_y
            else:
                # Post-yield behavior
                if v * F_d > 0:  # Loading
                    F_d_new = F_d
                else:  # Unloading
                    F_d_new = F_d + damper.k_bar * v * record.dt
                    if abs(F_d_new) > damper.F_y:
                        F_d_new = np.sign(F_d_new) * damper.F_y

            a = (
                -self.sys.k * u - self.sys.c * v - F_d - self.sys.m * ag * 9.81
            ) / self.sys.m

            return [v, a, (F_d_new - F_d) / record.dt]

        y0 = [0, 0, 0]
        sol = odeint(equations, y0, record.time)

        u = sol[:, 0]
        v = sol[:, 1]
        F_d = sol[:, 2]

        # Base shear
        F_base = self.sys.k * u + F_d

        # Energy calculations
        E_kinetic = 0.5 * self.sys.m * v**2
        E_elastic = 0.5 * self.sys.k * u**2

        # Damping energy
        E_damping = np.zeros_like(u)
        E_hysteretic = np.zeros_like(u)
        for i in range(1, len(u)):
            E_damping[i] = E_damping[i - 1] + self.sys.c * v[i] ** 2 * record.dt
            E_hysteretic[i] = E_hysteretic[i - 1] + F_d[i] * (u[i] - u[i - 1])

        # Input energy
        E_input = np.zeros_like(u)
        for i in range(1, len(u)):
            E_input[i] = E_input[i - 1] - self.sys.m * record.acc[i] * 9.81 * (
                u[i] - u[i - 1]
            )

        return {
            "time": record.time,
            "u": u,
            "v": v,
            "F_base": F_base,
            "F_damper": F_d,
            "u_max": np.max(np.abs(u)),
            "F_max": np.max(np.abs(F_base)),
            "E_kinetic": E_kinetic,
            "E_elastic": E_elastic,
            "E_damping": E_damping,
            "E_hysteretic": E_hysteretic,
            "E_input": E_input,
        }


def plot_response(results: Dict, title: str):
    """Plot system response"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14)

    # Displacement
    axes[0, 0].plot(results["time"], results["u"])
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Displacement (m)")
    axes[0, 0].set_title(f"Max = {results['u_max']:.4f} m")
    axes[0, 0].grid(True, alpha=0.3)

    # Base shear
    axes[0, 1].plot(results["time"], results["F_base"])
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Base Shear (N)")
    axes[0, 1].set_title(f"Max = {results['F_max']:.2f} N")
    axes[0, 1].grid(True, alpha=0.3)

    # Energy components
    axes[1, 0].plot(results["time"], results["E_input"], label="Input", linewidth=2)
    axes[1, 0].plot(results["time"], results["E_elastic"], label="Elastic")
    axes[1, 0].plot(results["time"], results["E_kinetic"], label="Kinetic")
    axes[1, 0].plot(results["time"], results["E_damping"], label="Damping")
    if "E_hysteretic" in results:
        axes[1, 0].plot(results["time"], results["E_hysteretic"], label="Hysteretic")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Energy (J)")
    axes[1, 0].set_title("Energy Components")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Energy balance
    E_total = results["E_elastic"] + results["E_kinetic"] + results["E_damping"]
    if "E_hysteretic" in results:
        E_total += results["E_hysteretic"]

    axes[1, 1].plot(results["time"], results["E_input"], label="Input", linewidth=2)
    axes[1, 1].plot(results["time"], E_total, label="Total (E+K+D+H)", linestyle="--")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Energy (J)")
    axes[1, 1].set_title("Energy Balance Check")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Main analysis routine"""

    # Initialize system
    system = SystemParams()
    analyzer = SDOFAnalysis(system)

    print(f"System Properties:")
    print(f"  Natural period T = {system.T} s")
    print(f"  Natural frequency ω = {system.omega:.2f} rad/s")
    print(f"  Stiffness k = {system.k:.2f} N/m")
    print(f"  Damping c = {system.c:.4f} N·s/m")
    print(f"  Damping ratio ζ = {system.zeta * 100}%\n")

    # Load seismic records
    records = {
        "El Centro": SeismicRecord("I-ELC180_AT2.txt").normalize(0.4),
        "Tabas": SeismicRecord("DAY-TR_AT2.txt").normalize(0.4),
    }

    # Step 1: Analyze elastic system
    print("=" * 60)
    print("STEP 1: ELASTIC SYSTEM ANALYSIS")
    print("=" * 60)

    elastic_results = {}
    F_bs_values = []

    for name, record in records.items():
        print(f"\n{name} Record:")
        result = analyzer.solve_elastic(record)
        elastic_results[name] = result
        F_bs_values.append(result["F_max"])

        print(f"  Max displacement: {result['u_max']:.4f} m")
        print(f"  Max base shear: {result['F_max']:.2f} N")

        fig = plot_response(result, f"Elastic System - {name}")
        plt.savefig(
            f"elastic_{name.replace(' ', '_')}.png", dpi=150, bbox_inches="tight"
        )

    F_bs = min(F_bs_values)
    print(f"\nF_bs (min of max base shears) = {F_bs:.2f} N")

    # Step 2: Analyze system with dampers
    print("\n" + "=" * 60)
    print("STEP 2: SYSTEM WITH DAMPERS")
    print("=" * 60)

    damper_ratios = [0.1, 0.5, 1.0]
    F_y = 0.4 * F_bs

    for ratio in damper_ratios:
        print(f"\n{'=' * 40}")
        print(f"Damper with k_bar = {ratio}k")
        print(f"  k_bar = {ratio * system.k:.2f} N/m")
        print(f"  F_y = {F_y:.2f} N")

        damper = DamperParams(k_bar=ratio * system.k, F_y=F_y)

        for name, record in records.items():
            print(f"\n  {name} Record:")
            result = analyzer.solve_with_damper(record, damper)

            print(f"    Max displacement: {result['u_max']:.4f} m")
            print(f"    Max base shear: {result['F_max']:.2f} N")

            # Calculate reduction ratios
            elastic_u = elastic_results[name]["u_max"]
            elastic_F = elastic_results[name]["F_max"]
            print(
                f"    Displacement reduction: {(1 - result['u_max'] / elastic_u) * 100:.1f}%"
            )
            print(
                f"    Base shear reduction: {(1 - result['F_max'] / elastic_F) * 100:.1f}%"
            )

            fig = plot_response(result, f"System with Damper (k_bar={ratio}k) - {name}")
            plt.savefig(
                f"damped_{ratio}k_{name.replace(' ', '_')}.png",
                dpi=150,
                bbox_inches="tight",
            )

    plt.show()


if __name__ == "__main__":
    main()
