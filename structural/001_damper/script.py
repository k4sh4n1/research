import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops


def read_seismic_record(filename):
    """Read seismic record with 4 header lines and variable data per line"""
    with open(filename, "r") as f:
        # Skip 4 header lines and extract dt and npts
        dt, npts = None, None
        for _ in range(4):
            header = f.readline()
            if "DT=" in header:
                dt_str = header.split("DT=")[1].split("SEC")[0]
                dt = float(dt_str.strip().replace(",", ""))
            if "NPTS=" in header:
                npts_str = header.split("NPTS=")[1]
                npts = int(npts_str.replace(",", "").split()[0])

        # Read all data values
        data = []
        for line in f:
            if not line.strip():
                continue
            values = line.strip().split()
            data.extend(float(v) for v in values)

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
    ops.model("basic", "-ndm", 1, "-ndf", 1)  # 1D model, 1 DOF

    # Create nodes
    ops.node(1, 0.0)  # Fixed base
    ops.node(2, 0.0)  # Free mass

    # Boundary conditions
    ops.fix(1, 1)  # Fix base node

    # Mass
    ops.mass(2, 1.0)  # Unit mass

    # Spring element
    ops.uniaxialMaterial("Elastic", 1, k)

    # Damper element (hysteretic)
    if with_damper:
        # Steel01 for elastic-plastic damper
        ops.uniaxialMaterial("Steel01", 2, Fy_bar, k_bar, 0.01)
        # Parallel combination of spring and damper
        ops.uniaxialMaterial("Parallel", 3, 1, 2)
        ops.element("zeroLength", 1, 1, 2, "-mat", 3, "-dir", 1)
    else:
        ops.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 1)

    # Rayleigh damping for inherent damping (5% damping ratio)
    omega = np.sqrt(k)  # natural frequency (m=1)
    a0 = 2 * omega * 0.05
    ops.rayleigh(a0, 0.0, 0.0, 0.0)


def run_analysis(acc, dt):
    """Run time history analysis"""
    # Define time series and load pattern
    ops.timeSeries("Path", 1, "-dt", dt, "-values", *acc.tolist())
    ops.pattern("UniformExcitation", 1, 1, "-accel", 1)

    # Analysis settings
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("BandGen")
    ops.test("NormDispIncr", 1.0e-8, 10)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    # Initialize output arrays
    time_vals = []
    disp = []
    vel = []
    accel = []
    base_reaction = []

    # Run analysis step by step
    for i in range(len(acc)):
        ok = ops.analyze(1, dt)
        if ok != 0:
            print(f"Analysis failed at step {i}")
            break

        time_vals.append(ops.getTime())
        disp.append(ops.nodeDisp(2, 1))
        vel.append(ops.nodeVel(2, 1))
        accel.append(ops.nodeAccel(2, 1))
        base_reaction.append(-ops.nodeReaction(1, 1))

    return (
        np.array(time_vals),
        np.array(disp),
        np.array(vel),
        np.array(accel),
        np.array(base_reaction),
    )


def compute_energies(
    time,
    disp,
    vel,
    acc_ground,
    k,
    c,
    with_damper=False,
    k_bar=0,
    Fy_bar=0,
    base_shear=None,
):
    """Compute energy components"""
    dt = time[1] - time[0] if len(time) > 1 else 0.01
    n = len(time)

    # Initialize energy arrays
    E_input = np.zeros(n)
    E_kinetic = np.zeros(n)
    E_elastic = np.zeros(n)
    E_damping = np.zeros(n)
    E_hysteretic = np.zeros(n)

    # For hysteretic damper tracking
    if with_damper and base_shear is not None:
        damper_disp = disp.copy()
        damper_force = np.zeros(n)
        yield_disp = Fy_bar / k_bar if k_bar > 0 else 0

        for i in range(n):
            # Estimate damper force from total base shear
            spring_force = k * disp[i]
            damper_force[i] = base_shear[i] - spring_force

    # Compute energies incrementally
    for i in range(1, n):
        # Current state energies
        E_kinetic[i] = 0.5 * 1.0 * vel[i] ** 2  # KE = 0.5*m*v^2
        E_elastic[i] = 0.5 * k * disp[i] ** 2  # PE = 0.5*k*x^2

        # Cumulative damping energy
        dE_damp = c * vel[i] ** 2 * dt
        E_damping[i] = E_damping[i - 1] + dE_damp

        # Cumulative hysteretic energy (if damper present)
        if with_damper and base_shear is not None:
            dE_hyst = damper_force[i] * (disp[i] - disp[i - 1])
            if dE_hyst > 0:  # Only dissipated energy
                E_hysteretic[i] = E_hysteretic[i - 1] + dE_hyst
            else:
                E_hysteretic[i] = E_hysteretic[i - 1]

        # Cumulative input energy
        dE_input = -1.0 * acc_ground[i] * vel[i] * dt  # m=1
        E_input[i] = E_input[i - 1] + dE_input

    return E_input, E_kinetic, E_elastic, E_damping, E_hysteretic


def main():
    # System properties
    T = 0.4  # Natural period (sec)
    zeta = 0.05  # Damping ratio
    m = 1.0  # Unit mass

    # Calculate stiffness and damping
    omega = 2 * np.pi / T
    k = omega**2 * m
    c = 2 * zeta * omega * m

    print(f"System properties:")
    print(f"  Period T = {T} sec")
    print(f"  Stiffness k = {k:.2f} N/m")
    print(f"  Damping c = {c:.4f} N·s/m")
    print(f"  Natural freq ω = {omega:.2f} rad/s")

    # Read seismic records
    records = []
    record_files = ["seismic1.txt", "seismic2.txt"]

    for i, filename in enumerate(record_files, 1):
        try:
            time, acc, dt = read_seismic_record(filename)
            acc_norm = normalize_record(acc, 0.4)
            records.append((time, acc_norm, dt, f"Record {i}"))
            print(f"\nRecord {i} loaded:")
            print(f"  Points: {len(acc)}")
            print(f"  Time step: {dt} sec")
            print(f"  Duration: {len(acc) * dt:.2f} sec")
            print(f"  Max acc: {np.max(np.abs(acc_norm)) / 9.81:.3f} g")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            # Create synthetic record for testing
            dt = 0.02 if i == 1 else 0.01
            duration = 20
            time = np.arange(0, duration, dt)
            freq = 2.0  # Hz
            acc_norm = (
                0.4 * 9.81 * np.sin(2 * np.pi * freq * time) * np.exp(-0.1 * time)
            )
            records.append((time, acc_norm, dt, f"Synthetic {i}"))
            print(f"Using synthetic record {i}")

    # Plot normalized records
    plt.figure(figsize=(12, 4))
    for time, acc, dt, label in records:
        plt.plot(time, acc / 9.81, label=label, alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (g)")
    plt.title("Normalized Seismic Records (PGA = 0.4g)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ANALYSIS 1: System without damper
    print("\n" + "=" * 60)
    print("ANALYSIS 1: System WITHOUT Damper")
    print("=" * 60)

    results_no_damper = []
    F_bs_values = []

    for time, acc, dt, label in records:
        print(f"\nAnalyzing {label}...")

        # Build and analyze
        build_sdof_model(k, c, with_damper=False)
        t, disp, vel, accel, base_shear = run_analysis(acc, dt)

        if len(disp) > 0:
            max_disp = np.max(np.abs(disp))
            max_shear = np.max(np.abs(base_shear))
            F_bs_values.append(max_shear)

            # Compute energies
            E_in, E_k, E_e, E_d, _ = compute_energies(t, disp, vel, acc[: len(t)], k, c)

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

    if not F_bs_values:
        print("Analysis failed - no results obtained")
        return

    F_bs = min(F_bs_values)
    print(f"\nF_bs = {F_bs:.2f} N (minimum of max base shears)")

    # Plot energy balance for system without damper
    for idx, res in enumerate(results_no_damper):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f"Energy Analysis - {res['label']} (No Damper)")

        # Energy components
        ax1.plot(res["time"], res["E_input"], "b-", label="Input", linewidth=2)
        ax1.plot(res["time"], res["E_kinetic"], "g-", label="Kinetic")
        ax1.plot(res["time"], res["E_damping"], "r-", label="Damping")
        ax1.plot(res["time"], res["E_elastic"], "m-", label="Elastic")
        ax1.set_ylabel("Energy (J)")
        ax1.set_title("Energy Components")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        # Energy balance check
        E_total = res["E_kinetic"] + res["E_damping"] + res["E_elastic"]
        ax2.plot(res["time"], res["E_input"], "b-", label="Input Energy", linewidth=2)
        ax2.plot(res["time"], E_total, "r--", label="Sum (K+D+E)", linewidth=1.5)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Energy (J)")
        ax2.set_title("Energy Balance Check")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ANALYSIS 2: System with dampers
    print("\n" + "=" * 60)
    print("ANALYSIS 2: System WITH Dampers")
    print("=" * 60)

    k_bar_ratios = [0.1, 0.5, 1.0]
    Fy_bar = 0.4 * F_bs

    print(f"\nDamper yield force: Fy_bar = {Fy_bar:.2f} N")

    for ratio in k_bar_ratios:
        k_bar = ratio * k
        print(f"\n" + "-" * 50)
        print(f"Damper stiffness: k_bar = {ratio}k = {k_bar:.2f} N/m")

        for idx, (time, acc, dt, label) in enumerate(records):
            print(f"\nAnalyzing {label} with damper...")

            # Build and analyze with damper
            build_sdof_model(k, c, with_damper=True, k_bar=k_bar, Fy_bar=Fy_bar)
            t, disp, vel, accel, base_shear = run_analysis(acc, dt)

            if len(disp) > 0:
                max_disp = np.max(np.abs(disp))
                max_shear = np.max(np.abs(base_shear))

                # Calculate reductions
                reduction_disp = 100 * (
                    1 - max_disp / results_no_damper[idx]["max_disp"]
                )
                reduction_shear = 100 * (
                    1 - max_shear / results_no_damper[idx]["max_shear"]
                )

                print(
                    f"  Max displacement: {max_disp:.4f} m (reduction: {reduction_disp:.1f}%)"
                )
                print(
                    f"  Max base shear: {max_shear:.2f} N (reduction: {reduction_shear:.1f}%)"
                )

                # Compute energy components including hysteretic energy
                E_in, E_k, E_e, E_d, E_h = compute_energies(
                    t,
                    disp,
                    vel,
                    acc[: len(t)],
                    k,
                    c,
                    with_damper=True,
                    k_bar=k_bar,
                    Fy_bar=Fy_bar,
                    base_shear=base_shear,
                )

                # Plot results
                fig, axs = plt.subplots(4, 1, figsize=(12, 10))
                fig.suptitle(f"System with Damper (k̄={ratio}k) – {label}")

                # (1) Compare displacement
                axs[0].plot(
                    results_no_damper[idx]["time"],
                    results_no_damper[idx]["disp"],
                    "b-",
                    label="No Damper",
                )
                axs[0].plot(t, disp, "r--", label="With Damper")
                axs[0].set_title("Displacement Comparison")
                axs[0].set_ylabel("Displacement (m)")
                axs[0].legend()
                axs[0].grid(True, alpha=0.3)

                # (2) Compare base shear
                axs[1].plot(
                    results_no_damper[idx]["time"],
                    results_no_damper[idx]["base_shear"],
                    "b-",
                    label="No Damper",
                )
                axs[1].plot(t, base_shear, "r--", label="With Damper")
                axs[1].set_title("Base Shear Comparison")
                axs[1].set_ylabel("Base Shear (N)")
                axs[1].legend()
                axs[1].grid(True, alpha=0.3)

                # (3) Energy Components
                axs[2].plot(t, E_in, "b-", label="Input", linewidth=2)
                axs[2].plot(t, E_k, "g-", label="Kinetic")
                axs[2].plot(t, E_e, "m-", label="Elastic")
                axs[2].plot(t, E_d, "r-", label="Damping")
                axs[2].plot(t, E_h, "c-", label="Hysteresis")
                axs[2].set_title("Energy Components with Damper")
                axs[2].set_ylabel("Energy (J)")
                axs[2].legend(loc="best")
                axs[2].grid(True, alpha=0.3)

                # (4) Energy balance
                E_total = E_k + E_e + E_d + E_h
                axs[3].plot(t, E_in, "b-", label="Input")
                axs[3].plot(t, E_total, "r--", label="Sum (K+D+E+H)")
                axs[3].set_title("Energy Balance Check")
                axs[3].set_xlabel("Time (s)")
                axs[3].set_ylabel("Energy (J)")
                axs[3].legend()
                axs[3].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()

    print("\nAll analyses completed successfully.\n")


if __name__ == "__main__":
    main()
