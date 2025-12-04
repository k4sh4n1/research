"""
Task 3: Seismic Response Analysis - Original Building vs Building with TMD
Compares 8th floor displacement, acceleration, and base shear for two earthquake records.
"""

import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops

# =============================================================================
# CONFIGURATION
# =============================================================================
NUM_STORIES = 8
MASS = 345.6  # tons
STIFFNESS = 3.404e5  # kN/m
DAMPING = 2937.0  # kN·s/m (equivalent to ton/s)

RECORD_FILES = {"El Centro": "record-ELCENTRO", "Tabas": "record-TABAS"}

G = 9.81  # m/s² → converts g to m/s², then to kN (since mass in tons)


# =============================================================================
# EARTHQUAKE RECORD PARSER
# =============================================================================
def read_peer_record(filename):
    """Parse PEER format ground motion file. Returns dt and acceleration in g."""
    with open(filename, "r") as f:
        lines = f.readlines()

    # Parse header line 4 for NPTS and DT
    header = lines[3]
    npts = int(header.split("NPTS=")[1].split(",")[0])
    dt = float(header.split("DT=")[1].split()[0])

    # Parse acceleration data (starts at line 5)
    accel = []
    for line in lines[4:]:
        accel.extend([float(x) for x in line.split()])

    return dt, np.array(accel[:npts])


# =============================================================================
# MODEL BUILDERS
# =============================================================================
def build_model_original():
    """Build 8-story shear building without TMD."""
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)

    # Nodes: 0 = base, 1-8 = floors
    for i in range(NUM_STORIES + 1):
        ops.node(i, 0.0)
    ops.fix(0, 1)

    # Mass
    for i in range(1, NUM_STORIES + 1):
        ops.mass(i, MASS)

    # Materials: stiffness + damping in parallel
    ops.uniaxialMaterial("Elastic", 1, STIFFNESS)
    ops.uniaxialMaterial("Viscous", 2, DAMPING, 1.0)  # c, alpha=1 (linear)
    ops.uniaxialMaterial("Parallel", 3, 1, 2)

    # Story elements
    for i in range(NUM_STORIES):
        ops.element("zeroLength", i + 1, i, i + 1, "-mat", 3, "-dir", 1)

    return NUM_STORIES  # roof node


def build_model_with_tmd(m_d, k_d, c_d):
    """Build 8-story shear building with TMD on roof."""
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)

    # Nodes: 0 = base, 1-8 = floors, 9 = TMD
    for i in range(NUM_STORIES + 2):
        ops.node(i, 0.0)
    ops.fix(0, 1)

    # Mass: floors + TMD
    for i in range(1, NUM_STORIES + 1):
        ops.mass(i, MASS)
    ops.mass(NUM_STORIES + 1, m_d)

    # Materials for building stories
    ops.uniaxialMaterial("Elastic", 1, STIFFNESS)
    ops.uniaxialMaterial("Viscous", 2, DAMPING, 1.0)
    ops.uniaxialMaterial("Parallel", 3, 1, 2)

    # Materials for TMD
    ops.uniaxialMaterial("Elastic", 4, k_d)
    ops.uniaxialMaterial("Viscous", 5, c_d, 1.0)
    ops.uniaxialMaterial("Parallel", 6, 4, 5)

    # Story elements
    for i in range(NUM_STORIES):
        ops.element("zeroLength", i + 1, i, i + 1, "-mat", 3, "-dir", 1)

    # TMD element (roof to TMD mass)
    ops.element(
        "zeroLength",
        NUM_STORIES + 1,
        NUM_STORIES,
        NUM_STORIES + 1,
        "-mat",
        6,
        "-dir",
        1,
    )

    return NUM_STORIES  # roof node


# =============================================================================
# TMD PARAMETER CALCULATOR
# =============================================================================
def calculate_tmd_params(mass_ratio=0.02):
    """Calculate TMD parameters based on first mode of original building."""
    # Build temporary model to get first mode properties
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    for i in range(NUM_STORIES + 1):
        ops.node(i, 0.0)
    ops.fix(0, 1)
    for i in range(1, NUM_STORIES + 1):
        ops.mass(i, MASS)
    ops.uniaxialMaterial("Elastic", 1, STIFFNESS)
    for i in range(NUM_STORIES):
        ops.element("zeroLength", i + 1, i, i + 1, "-mat", 1, "-dir", 1)

    eigenvalues = ops.eigen("-fullGenLapack", NUM_STORIES)
    omega1 = np.sqrt(eigenvalues[0])

    # First mode shape (normalized to roof = 1)
    phi1 = np.array([ops.nodeEigenvector(i, 1, 1) for i in range(1, NUM_STORIES + 1)])
    phi1 /= phi1[-1]

    # Generalized mass: M1 = Σ m_i * φ_i²
    M1 = MASS * np.sum(phi1**2)

    # TMD parameters
    m_d = mass_ratio * M1
    omega_d = omega1  # tuned to first mode
    k_d = m_d * omega_d**2
    c_d = 2 * 0.10 * m_d * omega_d  # 10% damping ratio

    ops.wipe()
    return m_d, k_d, c_d, M1, omega1


# =============================================================================
# TIME HISTORY ANALYSIS
# =============================================================================
def run_time_history(accel_g, dt, roof_node):
    """Run seismic analysis. Returns time, roof displacement, roof accel, base shear."""
    n_steps = len(accel_g)

    # Time series and load pattern
    ops.timeSeries("Path", 1, "-dt", dt, "-values", *accel_g, "-factor", G)
    ops.pattern("UniformExcitation", 1, 1, "-accel", 1)

    # Analysis settings
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("BandGen")
    ops.test("NormDispIncr", 1e-8, 100)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    # Storage arrays
    time_hist = []
    disp_roof = []
    accel_roof = []
    base_shear = []

    # Run analysis
    for _ in range(n_steps):
        ops.analyze(1, dt)
        time_hist.append(ops.getTime())
        disp_roof.append(ops.nodeDisp(roof_node, 1))
        accel_roof.append(
            ops.nodeAccel(roof_node, 1)
            + accel_g[min(len(time_hist) - 1, n_steps - 1)] * G
        )

        # Base shear = sum of inertia forces = Σ m_i * a_i (absolute)
        shear = sum(
            ops.nodeMass(i, 1)
            * (ops.nodeAccel(i, 1) + accel_g[min(len(time_hist) - 1, n_steps - 1)] * G)
            for i in range(1, ops.getNodeTags()[-1] + 1)
        )
        base_shear.append(-shear)

    return (
        np.array(time_hist),
        np.array(disp_roof),
        np.array(accel_roof),
        np.array(base_shear),
    )


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
def main():
    # Calculate TMD parameters (mass ratio = 0.02)
    m_d, k_d, c_d, M1, omega1 = calculate_tmd_params(mass_ratio=0.02)

    print("=" * 60)
    print("TMD PARAMETERS (m_d = 0.02 * M1)")
    print("=" * 60)
    print(f"  Generalized mass M1 = {M1:.2f} tons")
    print(f"  TMD mass m_d        = {m_d:.2f} tons")
    print(f"  TMD stiffness k_d   = {k_d:.2f} kN/m")
    print(f"  TMD damping c_d     = {c_d:.2f} kN·s/m")
    print(f"  First mode ω1       = {omega1:.4f} rad/s")
    print()

    results = {}

    for record_name, filename in RECORD_FILES.items():
        print(f"Processing: {record_name}")
        dt, accel_g = read_peer_record(filename)
        print(
            f"  NPTS = {len(accel_g)}, DT = {dt} s, Duration = {len(accel_g) * dt:.1f} s"
        )

        # --- Run for ORIGINAL building ---
        roof_node = build_model_original()
        t_orig, d_orig, a_orig, v_orig = run_time_history(accel_g, dt, roof_node)

        # --- Run for TMD-equipped building ---
        roof_node = build_model_with_tmd(m_d, k_d, c_d)
        t_tmd, d_tmd, a_tmd, v_tmd = run_time_history(accel_g, dt, roof_node)

        # Store results
        results[record_name] = {
            "time": t_orig,
            "disp_orig": d_orig,
            "disp_tmd": d_tmd,
            "accel_orig": a_orig,
            "accel_tmd": a_tmd,
            "shear_orig": v_orig,
            "shear_tmd": v_tmd,
        }

        # Peak response comparison
        max_d_orig, max_d_tmd = np.max(np.abs(d_orig)), np.max(np.abs(d_tmd))
        max_a_orig, max_a_tmd = np.max(np.abs(a_orig)), np.max(np.abs(a_tmd))
        max_v_orig, max_v_tmd = np.max(np.abs(v_orig)), np.max(np.abs(v_tmd))

        print(f"\n  {'Response':<25} {'Original':>12} {'With TMD':>12} {'Ratio':>10}")
        print(f"  {'-' * 60}")
        print(
            f"  {'Max Roof Disp (m)':<25} {max_d_orig:>12.4f} {max_d_tmd:>12.4f} {max_d_tmd / max_d_orig:>10.3f}"
        )
        print(
            f"  {'Max Roof Accel (m/s²)':<25} {max_a_orig:>12.4f} {max_a_tmd:>12.4f} {max_a_tmd / max_a_orig:>10.3f}"
        )
        print(
            f"  {'Max Base Shear (kN)':<25} {max_v_orig:>12.1f} {max_v_tmd:>12.1f} {max_v_tmd / max_v_orig:>10.3f}"
        )
        print()

    # ==========================================================================
    # PLOTTING
    # ==========================================================================
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex="col")

    for col, (record_name, res) in enumerate(results.items()):
        t = res["time"]

        # Row 0: Roof Displacement
        axes[0, col].plot(t, res["disp_orig"], "b-", lw=0.8, label="Original")
        axes[0, col].plot(t, res["disp_tmd"], "r-", lw=0.8, label="With TMD")
        axes[0, col].set_ylabel("Roof Disp. (m)")
        axes[0, col].set_title(f"{record_name} Earthquake")
        axes[0, col].legend(loc="upper right")
        axes[0, col].grid(True, alpha=0.3)

        # Row 1: Roof Acceleration
        axes[1, col].plot(t, res["accel_orig"], "b-", lw=0.8)
        axes[1, col].plot(t, res["accel_tmd"], "r-", lw=0.8)
        axes[1, col].set_ylabel("Roof Accel. (m/s²)")
        axes[1, col].grid(True, alpha=0.3)

        # Row 2: Base Shear
        axes[2, col].plot(t, res["shear_orig"], "b-", lw=0.8)
        axes[2, col].plot(t, res["shear_tmd"], "r-", lw=0.8)
        axes[2, col].set_ylabel("Base Shear (kN)")
        axes[2, col].set_xlabel("Time (s)")
        axes[2, col].grid(True, alpha=0.3)

    plt.suptitle(
        "Seismic Response Comparison: Original vs TMD-Equipped (m_d = 0.02·M₁)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("task4_response_comparison.png", dpi=150)
    plt.show()

    ops.wipe()


if __name__ == "__main__":
    main()
