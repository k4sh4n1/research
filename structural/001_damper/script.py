import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops


def read_seismic_record(filename):
    """Read seismic record with 4 header lines and variable data per line"""
    with open(filename, "r") as f:
        # Skip 4 header lines
        for _ in range(4):
            header = f.readline()
            if "DT=" in header:
                dt = float(header.split("DT=")[1].split("SEC")[0])
            if "NPTS=" in header:
                npts = int(header.split("NPTS=")[1].split()[0])

        # Read all data values
        data = []
        for line in f:
            values = line.strip().split()
            data.extend([float(v) for v in values])

    acc = np.array(data[:npts])
    time = np.arange(len(acc)) * dt
    return time, acc, dt


def normalize_record(acc, target_pga=0.4):
    """Normalize acceleration to target PGA in g"""
    max_acc = np.max(np.abs(acc))
    return acc * (target_pga * 9.81 / max_acc)


def build_sdof_model(k, c, with_damper=False, k_bar=0, Fy_bar=0):
    """Build SDOF model with or without hysteretic damper"""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    # Nodes
    ops.node(1, 0.0, 0.0)  # Fixed base
    ops.node(2, 0.0, 0.0)  # Mass node
    ops.fix(1, 1, 1, 1)

    # Mass (m = 1)
    ops.mass(2, 1.0, 0.0, 0.0)

    # Main spring
    ops.uniaxialMaterial("Elastic", 1, k)
    ops.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 1)

    # Damping
    ops.uniaxialMaterial("Viscous", 2, c, 1.0)
    ops.element("zeroLength", 2, 1, 2, "-mat", 2, "-dir", 1)

    # Hysteretic damper if needed
    if with_damper:
        # Steel01 for elastic-plastic behavior
        ops.uniaxialMaterial("Steel01", 3, Fy_bar, k_bar, 0.001)
        ops.element("zeroLength", 3, 1, 2, "-mat", 3, "-dir", 1)


def run_analysis(acc, dt):
    """Run time history analysis"""
    ops.timeSeries("Path", 1, "-dt", dt, "-values", *acc.tolist())
    ops.pattern("UniformExcitation", 1, 1, "-accel", 1)

    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("BandGen")
    ops.test("NormDispIncr", 1.0e-8, 10)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    # Initialize recorders
    time_vals = []
    disp = []
    vel = []
    accel = []
    base_shear = []

    # Run analysis
    for i in range(len(acc)):
        ops.analyze(1, dt)
        time_vals.append(ops.getTime())
        disp.append(ops.nodeDisp(2, 1))
        vel.append(ops.nodeVel(2, 1))
        accel.append(ops.nodeAccel(2, 1))
        base_shear.append(-ops.nodeReaction(1, 1))

    return (
        np.array(time_vals),
        np.array(disp),
        np.array(vel),
        np.array(accel),
        np.array(base_shear),
    )


def compute_energies(
    time, disp, vel, acc_total, k, c, k_bar=0, Fy_bar=0, with_damper=False
):
    """Compute energy components"""
    dt = time[1] - time[0]
    n = len(time)

    # Initialize energies
    E_input = np.zeros(n)
    E_kinetic = np.zeros(n)
    E_elastic = np.zeros(n)
    E_damping = np.zeros(n)
    E_hysteretic = np.zeros(n)

    # Compute incremental energies
    for i in range(1, n):
        # Kinetic energy
        E_kinetic[i] = 0.5 * vel[i] ** 2

        # Elastic energy
        E_elastic[i] = 0.5 * k * disp[i] ** 2

        # Damping energy (cumulative)
        E_damping[i] = E_damping[i - 1] + c * vel[i] ** 2 * dt

        # Input energy (cumulative)
        E_input[i] = E_input[i - 1] - acc_total[i] * vel[i] * dt

        # Hysteretic energy (simplified for damper)
        if with_damper and i > 1:
            # Approximate hysteretic energy dissipation
            d_disp = disp[i] - disp[i - 1]
            if abs(disp[i]) > Fy_bar / k_bar:  # Yielded
                E_hysteretic[i] = E_hysteretic[i - 1] + abs(Fy_bar * d_disp)

    return E_input, E_kinetic, E_elastic, E_damping, E_hysteretic


def main():
    # System properties
    T = 0.4  # Natural period (sec)
    zeta = 0.05  # Damping ratio
    m = 1.0  # Mass

    # Compute k and c
    omega = 2 * np.pi / T
    k = omega**2 * m
    c = 2 * zeta * omega * m

    print(f"System properties: k={k:.2f}, c={c:.4f}, ω={omega:.2f}")

    # Read and normalize seismic records
    records = []
    for i, filename in enumerate(["seismic1.txt", "seismic2.txt"], 1):
        try:
            time, acc, dt = read_seismic_record(filename)
            acc_norm = normalize_record(acc, 0.4)
            records.append((time, acc_norm, dt, f"Record {i}"))
            print(
                f"Record {i}: {len(acc)} points, dt={dt}s, max={np.max(np.abs(acc_norm)):.3f} m/s²"
            )
        except FileNotFoundError:
            print(f"Warning: {filename} not found. Using synthetic record.")
            # Create synthetic record for testing
            dt = 0.02 if i == 1 else 0.01
            time = np.arange(0, 20, dt)
            acc_norm = 0.4 * 9.81 * np.sin(2 * np.pi * time / T) * np.exp(-0.1 * time)
            records.append((time, acc_norm, dt, f"Synthetic {i}"))

    # Plot scaled records
    plt.figure(figsize=(10, 4))
    for time, acc, dt, label in records:
        plt.plot(time, acc / 9.81, label=label, alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (g)")
    plt.title("Normalized Seismic Records")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Analyze system without damper
    F_bs_values = []
    results_no_damper = []

    for time, acc, dt, label in records:
        print(f"\nAnalyzing {label} without damper...")
        build_sdof_model(k, c, with_damper=False)
        t, disp, vel, accel, base_shear = run_analysis(acc, dt)

        max_disp = np.max(np.abs(disp))
        max_shear = np.max(np.abs(base_shear))
        F_bs_values.append(max_shear)

        E_in, E_k, E_e, E_d, _ = compute_energies(t, disp, vel, acc, k, c)

        results_no_damper.append(
            {
                "label": label,
                "time": t,
                "disp": disp,
                "base_shear": base_shear,
                "max_disp": max_disp,
                "max_shear": max_shear,
                "E_input": E_in,
                "E_kinetic": E_k,
                "E_elastic": E_e,
                "E_damping": E_d,
            }
        )

        print(f"  Max displacement: {max_disp:.4f} m")
        print(f"  Max base shear: {max_shear:.2f} N")

    F_bs = min(F_bs_values)
    print(f"\nF_bs (min of max base shears): {F_bs:.2f} N")

    # Plot energy balance for system without damper
    for res in results_no_damper[:1]:  # Show first record only
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(res["time"], res["E_input"], label="Input", linewidth=2)
        ax1.plot(res["time"], res["E_kinetic"], label="Kinetic")
        ax1.plot(res["time"], res["E_damping"], label="Damping")
        ax1.plot(res["time"], res["E_elastic"], label="Elastic")
        ax1.set_ylabel("Energy (J)")
        ax1.set_title(f"Energy Components - {res['label']} (No Damper)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        E_total = res["E_kinetic"] + res["E_damping"] + res["E_elastic"]
        ax2.plot(res["time"], res["E_input"], label="Input Energy", linewidth=2)
        ax2.plot(res["time"], E_total, label="Sum (K+D+E)", linestyle="--")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Energy (J)")
        ax2.set_title("Energy Balance Check")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Analyze with dampers
    k_bar_ratios = [0.1, 0.5, 1.0]
    Fy_bar = 0.4 * F_bs

    for ratio in k_bar_ratios:
        k_bar = ratio * k
        print(f"\n{'=' * 50}")
        print(f"Analyzing with damper: k_bar = {ratio}k, Fy_bar = {Fy_bar:.2f} N")

        for idx, (time, acc, dt, label) in enumerate(
            records[:1]
        ):  # First record only for brevity
            print(f"\n{label}:")
            build_sdof_model(k, c, with_damper=True, k_bar=k_bar, Fy_bar=Fy_bar)
            t, disp, vel, accel, base_shear = run_analysis(acc, dt)

            max_disp = np.max(np.abs(disp))
            max_shear = np.max(np.abs(base_shear))

            print(
                f"  With damper: Max disp={max_disp:.4f} m, Max shear={max_shear:.2f} N"
            )
            print(
                f"  Reduction: Disp={100 * (1 - max_disp / results_no_damper[idx]['max_disp']):.1f}%, "
                + f"Shear={100 * (1 - max_shear / results_no_damper[idx]['max_shear']):.1f}%"
            )

            # Energy computation
            E_in, E_k, E_e, E_d, E_h = compute_energies(
                t, disp, vel, acc, k, c, k_bar, Fy_bar, True
            )

            # Plot comparison
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

            # Displacement comparison
            ax1.plot(
                results_no_damper[idx]["time"],
                results_no_damper[idx]["disp"],
                label="No damper",
                alpha=0.7,
            )
            ax1.plot(t, disp, label=f"With damper (k̄={ratio}k)", linewidth=2)
            ax1.set_ylabel("Displacement (m)")
            ax1.set_title("Displacement Response")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Base shear comparison
            ax2.plot(
                results_no_damper[idx]["time"],
                results_no_damper[idx]["base_shear"],
                label="No damper",
                alpha=0.7,
            )
            ax2.plot(t, base_shear, label=f"With damper (k̄={ratio}k)", linewidth=2)
            ax2.set_ylabel("Base Shear (N)")
            ax2.set_title("Base Shear Response")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Energy components
            ax3.plot(t, E_in, label="Input", linewidth=2)
            ax3.plot(t, E_k, label="Kinetic")
            ax3.plot(t, E_d, label="Damping")
            ax3.plot(t, E_e, label="Elastic")
            ax3.plot(t, E_h, label="Hysteretic", linestyle="--")
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Energy (J)")
            ax3.set_title("Energy Components")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Energy balance
            E_total = E_k + E_d + E_e + E_h
            ax4.plot(t, E_in, label="Input Energy", linewidth=2)
            ax4.plot(t, E_total, label="Sum (K+D+E+H)", linestyle="--")
            ax4.set_xlabel("Time (s)")
            ax4.set_ylabel("Energy (J)")
            ax4.set_title("Energy Balance Check")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.suptitle(f"{label} - Damper k̄={ratio}k", fontsize=14)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
