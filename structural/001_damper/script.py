import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops

# System parameters
m = 1.0  # Unit mass
T = 0.4  # Natural period (sec)
zeta = 0.05  # Damping ratio (5%)
g = 9.81  # Gravity acceleration (m/s^2)

# Derived parameters
omega = 2 * np.pi / T  # Natural frequency
k = omega**2 * m  # Stiffness
c = 2 * zeta * omega * m  # Damping coefficient


def read_seismic_record(filename, npts, dt):
    """Read seismic acceleration time history from file."""
    acc = []
    with open(filename, "r") as f:
        # Skip 4 header lines
        for _ in range(4):
            f.readline()
        # Read data line by line
        for line in f:
            values = line.strip().split()
            acc.extend([float(v) for v in values])

    acc = np.array(acc[:npts])
    # Normalize to 0.4g
    acc = acc * (0.4 * g / np.abs(acc).max())
    time = np.arange(npts) * dt
    return time, acc


def run_sdof_analysis(k, c, m, time, acc, damper_params=None):
    """Run SDOF analysis with or without damper using OpenSeesPy."""
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)

    # Create nodes
    ops.node(1, 0.0)  # Fixed base
    ops.node(2, 0.0)  # Mass node

    # Boundary conditions
    ops.fix(1, 1)

    # Mass
    ops.mass(2, m)

    # Spring and dashpot (main system)
    ops.uniaxialMaterial("Elastic", 1, k)
    ops.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 1)

    # Add damping
    ops.region(1, "-ele", 1)
    ops.rayleigh(0.0, 0.0, 0.0, 2 * zeta / omega)

    # Add hysteretic damper if specified
    if damper_params:
        k_bar, F_y = damper_params
        # Elastic-perfectly plastic material for damper
        ops.uniaxialMaterial("ElasticPP", 2, k_bar, F_y / k_bar)
        ops.element("zeroLength", 2, 1, 2, "-mat", 2, "-dir", 1)

    # Time series and load pattern
    ops.timeSeries("Path", 1, "-dt", time[1] - time[0], "-values", *acc, "-factor", 1.0)
    ops.pattern("UniformExcitation", 1, 1, "-accel", 1)

    # Analysis settings
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGen")
    ops.test("NormDispIncr", 1.0e-8, 10)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    # Initialize storage
    disp = []
    vel = []
    accel = []
    base_shear = []
    damper_force = [] if damper_params else None
    damper_disp = [] if damper_params else None

    # Run analysis
    for i, t in enumerate(time):
        ops.analyze(1, time[1] - time[0])

        disp.append(ops.nodeDisp(2, 1))
        vel.append(ops.nodeVel(2, 1))
        accel.append(ops.nodeAccel(2, 1))

        ops.reactions()
        base_shear.append(-ops.nodeReaction(1, 1))

        if damper_params:
            damper_force.append(ops.eleForce(2, 1))
            damper_disp.append(ops.nodeDisp(2, 1))

    results = {
        "time": time,
        "disp": np.array(disp),
        "vel": np.array(vel),
        "accel": np.array(accel),
        "base_shear": np.array(base_shear),
    }

    if damper_params:
        results["damper_force"] = np.array(damper_force)
        results["damper_disp"] = np.array(damper_disp)

    return results


def calculate_energies(results, k, c, m, acc):
    """Calculate energy components."""
    dt = results["time"][1] - results["time"][0]

    # Kinetic energy
    E_k = 0.5 * m * results["vel"] ** 2

    # Elastic energy (main spring only)
    E_e = 0.5 * k * results["disp"] ** 2

    # Damping energy (cumulative)
    E_d = np.cumsum(c * results["vel"] ** 2) * dt

    # Input energy (cumulative)
    E_i = -np.cumsum(m * acc * results["vel"]) * dt

    # Hysteretic energy if damper present
    E_h = None
    if "damper_force" in results:
        # Calculate incremental work and accumulate
        dE_h = results["damper_force"][:-1] * np.diff(results["damper_disp"])
        E_h = np.zeros(len(results["damper_force"]))
        E_h[1:] = np.cumsum(dE_h)

    return {
        "kinetic": E_k,
        "elastic": E_e,
        "damping": E_d,
        "input": E_i,
        "hysteretic": E_h,
    }


# Read seismic records
time1, acc1 = read_seismic_record("seismic1.txt", 1192, 0.02)
time2, acc2 = read_seismic_record("seismic2.txt", 4000, 0.01)

# System without damper
print("Analyzing system without damper...")
results1_no = run_sdof_analysis(k, c, m, time1, acc1)
results2_no = run_sdof_analysis(k, c, m, time2, acc2)

# Get F_bs (minimum of max base shears)
F_bs = min(
    np.abs(results1_no["base_shear"]).max(), np.abs(results2_no["base_shear"]).max()
)
print(f"F_bs = {F_bs:.3f} N")

# Energy calculations for system without damper
energy1_no = calculate_energies(results1_no, k, c, m, acc1)
energy2_no = calculate_energies(results2_no, k, c, m, acc2)

# Damper cases
damper_cases = [("k_bar = 0.1k", 0.1 * k), ("k_bar = 0.5k", 0.5 * k), ("k_bar = k", k)]

results_with_damper = {}
energies_with_damper = {}

for label, k_bar in damper_cases:
    print(f"\nAnalyzing system with damper: {label}")
    F_y = 0.4 * F_bs

    results1 = run_sdof_analysis(k, c, m, time1, acc1, (k_bar, F_y))
    results2 = run_sdof_analysis(k, c, m, time2, acc2, (k_bar, F_y))

    results_with_damper[label] = {"seismic1": results1, "seismic2": results2}

    energy1 = calculate_energies(results1, k, c, m, acc1)
    energy2 = calculate_energies(results2, k, c, m, acc2)

    energies_with_damper[label] = {"seismic1": energy1, "seismic2": energy2}

    print(f"  Max disp (Seismic 1): {np.abs(results1['disp']).max():.4f} m")
    print(f"  Max base shear (Seismic 1): {np.abs(results1['base_shear']).max():.2f} N")

# Plotting
print("\nGenerating plots...")

# 1. Scaled seismic records comparison
plt.figure(figsize=(10, 4))
plt.plot(time1, acc1 / g, "b-", label="Seismic 1", alpha=0.7)
plt.plot(time2, acc2 / g, "r-", label="Seismic 2", alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (g)")
plt.title("Scaled Seismic Records (0.4g peak)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 2. System without damper - energy plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Energy components
ax = axes[0]
ax.plot(time1, energy1_no["input"], "k-", label="Input")
ax.plot(time1, energy1_no["kinetic"], "b-", label="Kinetic")
ax.plot(time1, energy1_no["damping"], "r-", label="Damping")
ax.plot(time1, energy1_no["elastic"], "g-", label="Elastic")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Energy (J)")
ax.set_title("Energy Components (No Damper, Seismic 1)")
ax.legend()
ax.grid(True, alpha=0.3)

# Energy balance
ax = axes[1]
ax.plot(time1, energy1_no["input"], "k-", label="Input", linewidth=2)
total = energy1_no["kinetic"] + energy1_no["damping"] + energy1_no["elastic"]
ax.plot(time1, total, "r--", label="K+D+E", alpha=0.7)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Energy (J)")
ax.set_title("Energy Balance Check")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

# 3. System with damper - comparison plots
for label, k_bar in damper_cases:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    res_no = results1_no
    res_with = results_with_damper[label]["seismic1"]
    energy_with = energies_with_damper[label]["seismic1"]

    # Displacement comparison
    ax = axes[0, 0]
    ax.plot(time1, res_no["disp"], "b-", label="Without damper", alpha=0.7)
    ax.plot(time1, res_with["disp"], "r-", label="With damper", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (m)")
    ax.set_title(f"Displacement Comparison ({label})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Base shear comparison
    ax = axes[0, 1]
    ax.plot(time1, res_no["base_shear"], "b-", label="Without damper", alpha=0.7)
    ax.plot(time1, res_with["base_shear"], "r-", label="With damper", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Base Shear (N)")
    ax.set_title(f"Base Shear Comparison ({label})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Energy components
    ax = axes[1, 0]
    ax.plot(time1, energy_with["input"], "k-", label="Input")
    ax.plot(time1, energy_with["kinetic"], "b-", label="Kinetic")
    ax.plot(time1, energy_with["damping"], "r-", label="Damping")
    ax.plot(time1, energy_with["elastic"], "g-", label="Elastic")
    if energy_with["hysteretic"] is not None:
        ax.plot(time1, energy_with["hysteretic"], "m-", label="Hysteretic")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (J)")
    ax.set_title(f"Energy Components ({label})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Energy balance
    ax = axes[1, 1]
    ax.plot(time1, energy_with["input"], "k-", label="Input", linewidth=2)
    total = energy_with["kinetic"] + energy_with["damping"] + energy_with["elastic"]
    if energy_with["hysteretic"] is not None:
        total = total + energy_with["hysteretic"]
    ax.plot(time1, total, "r--", label="Sum", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (J)")
    ax.set_title("Energy Balance Check")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"System with Damper: {label}", fontsize=14)
    plt.tight_layout()

# 4. Hysteresis loops
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (label, k_bar) in enumerate(damper_cases):
    ax = axes[i]
    res = results_with_damper[label]["seismic1"]
    ax.plot(res["damper_disp"], res["damper_force"], "b-", linewidth=0.5)
    ax.set_xlabel("Displacement (m)")
    ax.set_ylabel("Damper Force (N)")
    ax.set_title(f"Hysteresis Loop ({label})")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)

plt.tight_layout()
plt.show()

# Performance summary
print("\n" + "=" * 50)
print("PERFORMANCE SUMMARY")
print("=" * 50)
print(f"\nSystem Properties:")
print(f"  Natural Period T = {T} s")
print(f"  Stiffness k = {k:.2f} N/m")
print(f"  Damping ratio ζ = {zeta * 100}%")
print(f"  F_bs = {F_bs:.3f} N")

print(f"\nMax Response Reduction (Seismic 1):")
for label, _ in damper_cases:
    disp_reduction = (
        1
        - np.abs(results_with_damper[label]["seismic1"]["disp"]).max()
        / np.abs(results1_no["disp"]).max()
    ) * 100
    shear_reduction = (
        1
        - np.abs(results_with_damper[label]["seismic1"]["base_shear"]).max()
        / np.abs(results1_no["base_shear"]).max()
    ) * 100
    print(f"  {label}:")
    print(f"    Max displacement reduction: {disp_reduction:.1f}%")
    print(f"    Max base shear reduction:   {shear_reduction:.1f}%")

print("\nAnalysis complete.")
