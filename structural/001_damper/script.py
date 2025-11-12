import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# System parameters
T = 0.4  # Natural period (sec)
zeta = 0.05  # Damping ratio
m = 1  # Unit mass
omega_n = 2 * np.pi / T  # Natural frequency
k = omega_n**2 * m  # Stiffness
c = 2 * zeta * omega_n * m  # Damping coefficient


# Load seismic records
def load_record(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        # Extract dt and npts from header
        for line in lines:
            if "NPTS=" in line and "DT=" in line:
                parts = line.split(",")
                npts = int(parts[0].split("=")[1])
                dt = float(parts[1].split("=")[1].split()[0])
                break
        # Extract acceleration data (skip header lines)
        data = []
        for line in lines[4:]:  # Skip first 4 header lines
            values = line.strip().split()
            data.extend([float(v) for v in values])
    return np.array(data[:npts]), dt


# Load and normalize records to 0.4g
ag1, dt1 = load_record("I-ELC180_AT2.txt")
ag2, dt2 = load_record("DAY-TR_AT2.txt")

# Normalize to 0.4g
ag1 = ag1 * (0.4 / np.max(np.abs(ag1)))
ag2 = ag2 * (0.4 / np.max(np.abs(ag2)))

# Create time arrays
t1 = np.arange(len(ag1)) * dt1
t2 = np.arange(len(ag2)) * dt2

# Convert to SI units (g to m/s²)
g = 9.81
ag1_si = ag1 * g
ag2_si = ag2 * g


# Newmark-Beta method for dynamic analysis
def newmark_beta(M, C, K, F, dt, beta=0.25, gamma=0.5):
    n = len(F)
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)

    # Initial acceleration
    a[0] = (F[0] - C * v[0] - K * u[0]) / M

    # Time stepping coefficients
    a1 = 1 / (beta * dt**2) * M + gamma / (beta * dt) * C
    a2 = 1 / (beta * dt) * M + (gamma / beta - 1) * C
    a3 = (1 / (2 * beta) - 1) * M + dt * (gamma / (2 * beta) - 1) * C
    K_eff = K + a1

    for i in range(n - 1):
        F_eff = F[i + 1] + a1 * u[i] + a2 * v[i] + a3 * a[i]
        u[i + 1] = F_eff / K_eff
        v[i + 1] = (
            gamma / (beta * dt) * (u[i + 1] - u[i])
            + (1 - gamma / beta) * v[i]
            + dt * (1 - gamma / (2 * beta)) * a[i]
        )
        a[i + 1] = (
            1 / (beta * dt**2) * (u[i + 1] - u[i])
            - 1 / (beta * dt) * v[i]
            - (1 / (2 * beta) - 1) * a[i]
        )

    return u, v, a


# Analyze system without damper
def analyze_system(ag, dt, label):
    F = -m * ag  # Force array
    u, v, a = newmark_beta(m, c, k, F, dt)

    # Base shear
    Fb = k * u + c * v

    # Energy calculations
    E_kinetic = 0.5 * m * v**2
    E_elastic = 0.5 * k * u**2
    E_damping = np.cumsum(c * v**2 * dt)
    E_input = -np.cumsum(ag * v * m * dt)

    print(f"\n{label}:")
    print(f"  Max displacement: {np.max(np.abs(u)):.4f} m")
    print(f"  Max base shear: {np.max(np.abs(Fb)):.4f} N")

    return u, v, a, Fb, E_kinetic, E_elastic, E_damping, E_input


# System alone analysis
print("=" * 50)
print("SYSTEM ALONE (No Damper)")
print("=" * 50)

u1, v1, a1, Fb1, Ek1, Ee1, Ed1, Ei1 = analyze_system(
    ag1_si, dt1, "Record 1 (El Centro)"
)
u2, v2, a2, Fb2, Ek2, Ee2, Ed2, Ei2 = analyze_system(ag2_si, dt2, "Record 2 (Tabas)")

# Store Fbs as minimum of max base shears
Fbs = min(np.max(np.abs(Fb1)), np.max(np.abs(Fb2)))
print(f"\nFbs (min of max base shears): {Fbs:.4f} N")

# Plot energy comparison for system alone
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("System Energy Analysis (No Damper)")

# Record 1
axes[0, 0].plot(t1, Ei1, label="Input")
axes[0, 0].plot(t1, Ed1, label="Damping")
axes[0, 0].plot(t1, Ee1, label="Elastic")
axes[0, 0].plot(t1, Ek1, label="Kinetic")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Energy (J)")
axes[0, 0].set_title("Record 1 - Energy Components")
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(t1, Ei1, label="Input", linewidth=2)
axes[0, 1].plot(t1, Ed1 + Ee1 + Ek1, "--", label="Sum(D+E+K)")
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Cumulative Energy (J)")
axes[0, 1].set_title("Record 1 - Energy Balance")
axes[0, 1].legend()
axes[0, 1].grid(True)

# Record 2
axes[1, 0].plot(t2, Ei2, label="Input")
axes[1, 0].plot(t2, Ed2, label="Damping")
axes[1, 0].plot(t2, Ee2, label="Elastic")
axes[1, 0].plot(t2, Ek2, label="Kinetic")
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("Energy (J)")
axes[1, 0].set_title("Record 2 - Energy Components")
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(t2, Ei2, label="Input", linewidth=2)
axes[1, 1].plot(t2, Ed2 + Ee2 + Ek2, "--", label="Sum(D+E+K)")
axes[1, 1].set_xlabel("Time (s)")
axes[1, 1].set_ylabel("Cumulative Energy (J)")
axes[1, 1].set_title("Record 2 - Energy Balance")
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()


# System with damper - Bilinear hysteretic model
def analyze_with_damper(ag, dt, k_bar_ratio, label):
    k_bar = k_bar_ratio * k
    Fy = 0.4 * Fbs

    n = len(ag)
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    Fd = np.zeros(n)  # Damper force

    # Hysteresis state
    u_yield = Fy / k_bar
    plastic_disp = 0
    yielded = False
    last_dir = 0

    # Initial conditions
    F = -m * ag[0]
    a[0] = F / m

    # Time stepping
    for i in range(n - 1):
        F = -m * ag[i + 1]

        # Predictor
        u_pred = u[i] + dt * v[i] + 0.5 * dt**2 * a[i]
        v_pred = v[i] + dt * a[i]

        # Damper force calculation (bilinear)
        du = u_pred - u[i]

        if not yielded:
            Fd_trial = Fd[i] + k_bar * du
            if abs(Fd_trial) >= Fy:
                yielded = True
                Fd[i + 1] = Fy * np.sign(Fd_trial)
                plastic_disp = u_pred - Fd[i + 1] / k_bar
                last_dir = np.sign(du)
            else:
                Fd[i + 1] = Fd_trial
        else:
            # Check for unloading
            if np.sign(du) != last_dir:
                yielded = False
                Fd[i + 1] = Fd[i] + k_bar * du
                if abs(Fd[i + 1]) >= Fy:
                    yielded = True
                    Fd[i + 1] = Fy * np.sign(Fd[i + 1])
                    last_dir = np.sign(du)
            else:
                Fd[i + 1] = Fd[i]  # Perfect plasticity

        # Corrector
        a[i + 1] = (F - c * v_pred - k * u_pred - Fd[i + 1]) / m
        v[i + 1] = v_pred + 0.5 * dt * (a[i + 1] - a[i])
        u[i + 1] = u_pred + dt**2 / 6 * (a[i + 1] - a[i])

    # Calculate energies
    Fb = k * u + c * v + Fd
    E_kinetic = 0.5 * m * v**2
    E_elastic = 0.5 * k * u**2
    E_damping = np.zeros(n)
    E_hysteretic = np.zeros(n)

    for i in range(1, n):
        E_damping[i] = E_damping[i - 1] + c * v[i] ** 2 * dt
        E_hysteretic[i] = E_hysteretic[i - 1] + Fd[i] * (u[i] - u[i - 1])

    E_input = -np.cumsum(ag * v * m * dt)

    print(f"\n{label} with k̄/k = {k_bar_ratio}:")
    print(f"  Max displacement: {np.max(np.abs(u)):.4f} m")
    print(f"  Max base shear: {np.max(np.abs(Fb)):.4f} N")

    return u, v, Fb, E_kinetic, E_elastic, E_damping, E_hysteretic, E_input


# Analyze with three different damper stiffnesses
print("\n" + "=" * 50)
print("SYSTEM WITH DAMPER")
print("=" * 50)

k_bar_ratios = [0.1, 0.5, 1.0]

for k_ratio in k_bar_ratios:
    print(f"\n--- k̄/k = {k_ratio} ---")

    u1d, v1d, Fb1d, Ek1d, Ee1d, Ed1d, Eh1d, Ei1d = analyze_with_damper(
        ag1_si, dt1, k_ratio, "Record 1"
    )
    u2d, v2d, Fb2d, Ek2d, Ee2d, Ed2d, Eh2d, Ei2d = analyze_with_damper(
        ag2_si, dt2, k_ratio, "Record 2"
    )

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Energy Analysis with Damper (k̄/k = {k_ratio})")

    axes[0].plot(t1, Ei1d, label="Input", linewidth=2)
    axes[0].plot(t1, Ed1d, label="Viscous Damping")
    axes[0].plot(t1, Eh1d, label="Hysteretic")
    axes[0].plot(t1, Ee1d, label="Elastic")
    axes[0].plot(t1, Ek1d, label="Kinetic")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Energy (J)")
    axes[0].set_title("Record 1")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t2, Ei2d, label="Input", linewidth=2)
    axes[1].plot(t2, Ed2d, label="Viscous Damping")
    axes[1].plot(t2, Eh2d, label="Hysteretic")
    axes[1].plot(t2, Ee2d, label="Elastic")
    axes[1].plot(t2, Ek2d, label="Kinetic")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Energy (J)")
    axes[1].set_title("Record 2")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
