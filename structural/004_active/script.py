"""
Active Structural Control - Series 4 (Final Version)
LQR and Instantaneous Optimal Control for 8-story shear building.
Both methods tuned to achieve 25% target.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.linalg import expm, solve_continuous_are

# =============================================================================
# CONFIGURATION
# =============================================================================
NUM_STORIES = 8
MASS = 345.6  # tons
STIFFNESS = 3.404e5  # kN/m
DAMPING = 2937.0  # kN·s/m
G = 9.81  # m/s²

RECORD_FILES = {"El Centro": "record-ELCENTRO", "Tabas": "record-TABAS"}
TARGET_REDUCTION = 0.25


# =============================================================================
# EARTHQUAKE PARSER
# =============================================================================
def read_peer_record(filename):
    """Parse PEER format ground motion file."""
    with open(filename, "r") as f:
        lines = f.readlines()
    header = lines[3]
    npts = int(header.split("NPTS=")[1].split(",")[0])
    dt = float(header.split("DT=")[1].split()[0])
    accel = []
    for line in lines[4:]:
        accel.extend(map(float, line.split()))
    return dt, np.array(accel[:npts])


# =============================================================================
# STATE-SPACE MODEL
# =============================================================================
def build_state_space(n, m, k, c):
    """
    Build state-space matrices for n-story shear building.
    State: z = [x1, ..., xn, ẋ1, ..., ẋn]^T

    Uses inter-story actuators: actuator i connects floor i-1 to floor i
    (floor 0 = ground).
    """
    M = m * np.eye(n)

    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = k if i == n - 1 else 2 * k
        if i > 0:
            K[i, i - 1] = -k
        if i < n - 1:
            K[i, i + 1] = -k

    C = (c / k) * K
    M_inv = np.linalg.inv(M)

    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -M_inv @ K
    A[n:, n:] = -M_inv @ C

    # Inter-story actuator influence matrix
    # Actuator i acts between floor i-1 and floor i
    # Net force on floor i: +u_i - u_{i+1} (Newton's 3rd law)
    D = np.eye(n) - np.diag(np.ones(n - 1), k=1)

    Bu = np.zeros((2 * n, n))
    Bu[n:, :] = M_inv @ D

    Br = np.zeros((2 * n, 1))
    Br[n:, 0] = -np.ones(n)

    return A, Bu, Br, M, K, C


# =============================================================================
# DISCRETIZATION
# =============================================================================
def discretize_system(A, Bu, Br, dt):
    """Exact discretization using matrix exponential."""
    n = A.shape[0]
    n_u = Bu.shape[1]
    n_w = Br.shape[1]

    M_aug = np.zeros((n + n_u + n_w, n + n_u + n_w))
    M_aug[:n, :n] = A * dt
    M_aug[:n, n : n + n_u] = Bu * dt
    M_aug[:n, n + n_u :] = Br * dt

    exp_M = expm(M_aug)

    Ad = exp_M[:n, :n]  # = e^(A*dt)
    Bd = exp_M[:n, n : n + n_u]  # = ∫ e^(A(dt-τ)) Bu dτ
    Ed = exp_M[:n, n + n_u :]  # = ∫ e^(A(dt-τ)) Br dτ

    return Ad, Bd, Ed


# =============================================================================
# LQR CONTROL
# =============================================================================
def compute_lqr_gain(A, Bu, Q, R):
    """Solve continuous-time Riccati equation."""
    P = solve_continuous_are(A, Bu, Q, R)
    K = np.linalg.solve(R, Bu.T @ P)
    return K, P


def simulate_lqr(A, Bu, Br, K, z0, accel_g, dt):
    """Simulate closed-loop system with LQR: u = -Kz"""
    n_steps = len(accel_g)
    t = np.arange(n_steps) * dt
    A_cl = A - Bu @ K

    def dzdt(z, ti):
        idx = min(int(ti / dt), n_steps - 1)
        return A_cl @ z + Br.flatten() * accel_g[idx] * G

    z = odeint(dzdt, z0, t)
    u = -z @ K.T
    return t, z, u


# =============================================================================
# INSTANTANEOUS OPTIMAL CONTROL
# =============================================================================


# Instability is caused by this method, so we won't use it.
def simulate_instantaneous_textbook(A, Bu, Br, Q, R, z0, accel_g, dt):
    """
    Instantaneous optimal control - Textbook Version.
    Control law: u = -(dt/2) * R^(-1) * B^T * Q * z
    """
    n_steps = len(accel_g)
    n_states = A.shape[0]
    n_control = Bu.shape[1]

    t = np.arange(n_steps) * dt
    z = np.zeros((n_steps, n_states))
    u = np.zeros((n_steps, n_control))
    z[0] = z0

    # Textbook gain: K = (dt/2) * R^(-1) * B^T * Q
    R_inv = np.linalg.inv(R)
    K_textbook = (dt / 2) * R_inv @ Bu.T @ Q

    # Use continuous dynamics for simulation (Euler integration)
    for i in range(n_steps - 1):
        w_k = accel_g[i] * G

        # Textbook control law
        u[i] = -K_textbook @ z[i]

        # Euler integration: z_{k+1} = z_k + dt * (A*z + Bu*u + Br*w)
        dz = A @ z[i] + Bu @ u[i] + Br.flatten() * w_k
        z[i + 1] = z[i] + dt * dz

        if np.any(np.abs(z[i + 1]) > 1e10):
            print(f"  Warning: Instability at step {i}")
            z[i + 1 :] = np.nan
            u[i + 1 :] = np.nan
            break

    return t, z, u


# =============================================================================
# INSTANTANEOUS OPTIMAL CONTROL - DERIVATION
# =============================================================================
#
# DISCRETE-TIME STATE EQUATION:
# -----------------------------
# Using exact discretization (matrix exponential), the state equation is:
#
#   z_{k+1} = A_d * z_k + B_d * u_k + E_d * w_k
#
# where:
#   z_k     = state vector [displacements; velocities] at step k
#   u_k     = control force vector at step k
#   w_k     = ground acceleration at step k (disturbance)
#   A_d     = discrete state matrix (= e^{A*dt})
#   B_d     = discrete control input matrix
#   E_d     = discrete disturbance input matrix
#
# COST FUNCTION:
# --------------
# At each time step, minimize the instantaneous cost:
#
#   J_k = z_{k+1}^T * Q * z_{k+1} + u_k^T * R * u_k
#
# where:
#   Q = state weighting matrix (penalizes displacements/velocities)
#   R = control weighting matrix (penalizes control effort)
#
# OPTIMIZATION:
# -------------
# Substitute the state equation into J_k:
#
#   J_k = (A_d*z_k + B_d*u_k + E_d*w_k)^T * Q * (A_d*z_k + B_d*u_k + E_d*w_k)
#         + u_k^T * R * u_k
#
# Take the derivative with respect to u_k and set equal to zero:
#
#   dJ_k/du_k = 2 * B_d^T * Q * (A_d*z_k + B_d*u_k + E_d*w_k) + 2 * R * u_k = 0
#
# Rearranging:
#
#   B_d^T * Q * B_d * u_k + R * u_k = -B_d^T * Q * A_d * z_k - B_d^T * Q * E_d * w_k
#
#   (B_d^T * Q * B_d + R) * u_k = -B_d^T * Q * A_d * z_k - B_d^T * Q * E_d * w_k
#
# OPTIMAL CONTROL LAW:
# --------------------
# Solving for u_k:
#
#   u_k = -(B_d^T * Q * B_d + R)^{-1} * B_d^T * Q * A_d * z_k
#         -(B_d^T * Q * B_d + R)^{-1} * B_d^T * Q * E_d * w_k
#
# Define the gain matrices:
#
#   K_1 = (B_d^T * Q * B_d + R)^{-1} * B_d^T * Q * A_d    (feedback gain)
#   K_2 = (B_d^T * Q * B_d + R)^{-1} * B_d^T * Q * E_d    (feedforward gain)
#
# Full control law:
#
#   u_k = -K_1 * z_k - K_2 * w_k
#
# PURE FEEDBACK VERSION:
# ----------------------
# If ground acceleration w_k is unknown or unreliable, use feedback only:
#
#   u_k = -K_1 * z_k
#
# This is the version implemented below.
#
# =============================================================================


def simulate_instantaneous(Ad, Bd, Ed, Q, R, z0, accel_g, dt):
    """
    Instantaneous Optimal Control (Discrete-Time Formulation).

    Minimizes at each step:  J_k = z_{k+1}^T Q z_{k+1} + u_k^T R u_k

    Optimal control law:     u_k = -K_1 @ z_k

    where: K_1 = (B_d^T Q B_d + R)^{-1} B_d^T Q A_d
    """
    n_steps = len(accel_g)
    n_states = Ad.shape[0]
    n_control = Bd.shape[1]

    t = np.arange(n_steps) * dt
    z = np.zeros((n_steps, n_states))
    u = np.zeros((n_steps, n_control))
    z[0] = z0

    # Compute feedback gain K_1
    # K_1 = (B_d^T Q B_d + R)^{-1} * B_d^T Q A_d
    BQB_R = Bd.T @ Q @ Bd + R  # (B_d^T Q B_d + R)
    K1 = np.linalg.solve(BQB_R, Bd.T @ Q @ Ad)  # More stable than explicit inverse

    for i in range(n_steps - 1):
        w_k = accel_g[i] * G

        # Optimal control (feedback only)
        u[i] = -K1 @ z[i]

        # State update
        z[i + 1] = Ad @ z[i] + Bd @ u[i] + Ed.flatten() * w_k

        # Check for numerical instability
        if np.any(np.abs(z[i + 1]) > 1e10):
            print(f"  Warning: Instability at step {i}")
            z[i + 1 :] = np.nan
            u[i + 1 :] = np.nan
            break

    return t, z, u


# =============================================================================
# UNCONTROLLED SIMULATION
# =============================================================================
def simulate_uncontrolled(A, Br, z0, accel_g, dt):
    """Simulate uncontrolled system."""
    n_steps = len(accel_g)
    t = np.arange(n_steps) * dt

    def dzdt(z, ti):
        idx = min(int(ti / dt), n_steps - 1)
        return A @ z + Br.flatten() * accel_g[idx] * G

    z = odeint(dzdt, z0, t)
    return t, z


# =============================================================================
# Q-MATRIX TUNING FOR LQR
# =============================================================================
def make_Q_lqr(alpha, n):
    """Q matrix for LQR: weight displacements more than velocities."""
    Q = np.zeros((2 * n, 2 * n))
    Q[:n, :n] = alpha * np.eye(n)
    Q[n:, n:] = alpha * 0.01 * np.eye(n)
    return Q


def tune_lqr(A, Bu, Br, z0, accel_g, dt, target_ratio):
    """Find Q scaling for LQR to achieve target reduction."""
    n = NUM_STORIES
    R = np.eye(n)

    _, z_unc = simulate_uncontrolled(A, Br, z0, accel_g, dt)
    max_unc = np.max(np.abs(z_unc[:, n - 1]))

    print(f"  Uncontrolled max roof disp: {max_unc:.4f} m")
    print(f"  Target ({target_ratio:.0%}): {target_ratio * max_unc:.4f} m")

    alpha_low, alpha_high = 1e-2, 1e14
    best_alpha, best_K, best_ratio = alpha_low, None, 1.0

    for _ in range(80):
        alpha = np.sqrt(alpha_low * alpha_high)
        Q = make_Q_lqr(alpha, n)

        try:
            K, _ = compute_lqr_gain(A, Bu, Q, R)
            _, z_ctrl, _ = simulate_lqr(A, Bu, Br, K, z0, accel_g, dt)
            max_ctrl = np.max(np.abs(z_ctrl[:, n - 1]))
            ratio = max_ctrl / max_unc

            if abs(ratio - target_ratio) < 0.005:
                print(f"  LQR: Found α = {alpha:.2e} → ratio = {ratio:.1%}")
                return alpha, K, Q, max_unc
            elif ratio > target_ratio:
                alpha_low = alpha
            else:
                alpha_high = alpha

            if abs(ratio - target_ratio) < abs(best_ratio - target_ratio):
                best_alpha, best_K, best_ratio = alpha, K, ratio

        except Exception:
            alpha_high = alpha

    print(f"  LQR Best: α = {best_alpha:.2e} → ratio = {best_ratio:.1%}")
    return best_alpha, best_K, make_Q_lqr(best_alpha, n), max_unc


# =============================================================================
# Q-MATRIX TUNING FOR INSTANTANEOUS
# =============================================================================
def make_Q_inst(alpha, n):
    """Q matrix for Instantaneous: same structure as LQR."""
    Q = np.zeros((2 * n, 2 * n))
    Q[:n, :n] = alpha * np.eye(n)
    Q[n:, n:] = alpha * 0.01 * np.eye(n)
    return Q


def tune_instantaneous(Ad, Bd, Ed, z0, accel_g, dt, target_ratio, max_unc):
    """Find Q scaling for Instantaneous to achieve target reduction."""
    n = NUM_STORIES
    R = np.eye(n)  # R = I as required by assignment

    alpha_low, alpha_high = 1e0, 1e20
    best_alpha, best_Q, best_ratio = alpha_low, None, 1.0

    for _ in range(100):
        alpha = np.sqrt(alpha_low * alpha_high)
        Q = make_Q_inst(alpha, n)

        try:
            _, z_ctrl, _ = simulate_instantaneous(Ad, Bd, Ed, Q, R, z0, accel_g, dt)
            max_ctrl = np.nanmax(np.abs(z_ctrl[:, n - 1]))

            if np.isnan(max_ctrl) or max_ctrl > 1e5:
                alpha_high = alpha
                continue

            ratio = max_ctrl / max_unc

            if abs(ratio - target_ratio) < 0.01:
                print(f"  Inst: Found α = {alpha:.2e} → ratio = {ratio:.1%}")
                return Q, R
            elif ratio > target_ratio:
                alpha_low = alpha
            else:
                alpha_high = alpha

            if abs(ratio - target_ratio) < abs(best_ratio - target_ratio):
                best_alpha, best_Q, best_ratio = alpha, Q, ratio

        except Exception:
            alpha_high = alpha

    print(f"  Inst Best: α = {best_alpha:.2e} → ratio = {best_ratio:.1%}")
    return make_Q_inst(best_alpha, n), R


# =============================================================================
# MAIN
# =============================================================================
def main():
    A, Bu, Br, M, K_mat, C = build_state_space(NUM_STORIES, MASS, STIFFNESS, DAMPING)
    z0 = np.zeros(2 * NUM_STORIES)
    n = NUM_STORIES

    print("=" * 70)
    print("ACTIVE STRUCTURAL CONTROL - 8-STORY BUILDING (FINAL)")
    print("=" * 70)

    results = {}

    for rec_name, filename in RECORD_FILES.items():
        print(f"\n{'─' * 70}")
        print(f"Earthquake: {rec_name}")
        print("─" * 70)

        dt, accel_g = read_peer_record(filename)
        print(
            f"  Record: {len(accel_g)} pts, dt={dt}s, duration={len(accel_g) * dt:.1f}s"
        )

        Ad, Bd, Ed = discretize_system(A, Bu, Br, dt)

        # Tune LQR
        print("\nTuning LQR...")
        alpha_lqr, K_lqr, Q_lqr, max_unc = tune_lqr(
            A, Bu, Br, z0, accel_g, dt, TARGET_REDUCTION
        )

        # Tune Instantaneous separately
        print("\nTuning Instantaneous...")
        Q_inst, R_inst = tune_instantaneous(
            Ad, Bd, Ed, z0, accel_g, dt, TARGET_REDUCTION, max_unc
        )

        # Simulate all
        t, z_unc = simulate_uncontrolled(A, Br, z0, accel_g, dt)
        t, z_lqr, u_lqr = simulate_lqr(A, Bu, Br, K_lqr, z0, accel_g, dt)
        t, z_inst, u_inst = simulate_instantaneous(
            Ad, Bd, Ed, Q_inst, R_inst, z0, accel_g, dt
        )

        # Compute absolute accelerations
        def get_abs_accel(z, accel_g, dt):
            acc = np.zeros(len(z))
            for i in range(len(z)):
                idx = min(i, len(accel_g) - 1)
                # Roof acceleration = relative + ground
                acc[i] = (A[n + n - 1, :] @ z[i]) + accel_g[idx] * G
            return acc

        accel_unc = get_abs_accel(z_unc, accel_g, dt)
        accel_lqr = get_abs_accel(z_lqr, accel_g, dt)
        accel_inst = get_abs_accel(z_inst, accel_g, dt)

        # Base shear (sum of inertia forces)
        def get_base_shear(z, accel_g, dt):
            shear = np.zeros(len(z))
            for i in range(len(z)):
                idx = min(i, len(accel_g) - 1)
                # Base shear = sum of m_j * (relative_accel_j + ground_accel)
                for j in range(n):
                    rel_accel = A[n + j, :] @ z[i]
                    shear[i] += MASS * (rel_accel + accel_g[idx] * G)
            return -shear

        shear_unc = get_base_shear(z_unc, accel_g, dt)
        shear_lqr = get_base_shear(z_lqr, accel_g, dt)
        shear_inst = get_base_shear(z_inst, accel_g, dt)

        results[rec_name] = {
            "t": t,
            "dt": dt,
            "z_unc": z_unc,
            "z_lqr": z_lqr,
            "z_inst": z_inst,
            "u_lqr": u_lqr,
            "u_inst": u_inst,
            "accel_unc": accel_unc,
            "accel_lqr": accel_lqr,
            "accel_inst": accel_inst,
            "shear_unc": shear_unc,
            "shear_lqr": shear_lqr,
            "shear_inst": shear_inst,
        }

        # Print summary
        d_unc = np.max(np.abs(z_unc[:, n - 1]))
        d_lqr = np.max(np.abs(z_lqr[:, n - 1]))
        d_inst = np.nanmax(np.abs(z_inst[:, n - 1]))

        print(f"\n  {'Response':<25} {'Uncontrol':>12} {'LQR':>12} {'Instant':>12}")
        print(f"  {'-' * 55}")
        print(
            f"  {'Max Roof Disp (m)':<25} {d_unc:>12.4f} {d_lqr:>12.4f} {d_inst:>12.4f}"
        )
        print(
            f"  {'Ratio':<25} {'100%':>12} {d_lqr / d_unc:>11.1%} {d_inst / d_unc:>11.1%}"
        )
        print(
            f"  {'Max Ctrl Force (kN)':<25} {'N/A':>12} {np.max(np.abs(u_lqr)):>12.1f} {np.nanmax(np.abs(u_inst)):>12.1f}"
        )

    plot_results(results)
    print_full_table(results)


def plot_results(results):
    """Generate separate time history plots for each response type."""
    n = NUM_STORIES

    plot_configs = [
        (
            "roof_displacement",
            "Roof Displacement (m)",
            lambda r: r["z_unc"][:, n - 1],
            lambda r: r["z_lqr"][:, n - 1],
            lambda r: r["z_inst"][:, n - 1],
        ),
        (
            "roof_acceleration",
            "Roof Acceleration (m/s²)",
            lambda r: r["accel_unc"],
            lambda r: r["accel_lqr"],
            lambda r: r["accel_inst"],
        ),
        (
            "base_shear",
            "Base Shear (MN)",
            lambda r: r["shear_unc"] / 1000,
            lambda r: r["shear_lqr"] / 1000,
            lambda r: r["shear_inst"] / 1000,
        ),
    ]

    # Plot response comparisons (Uncontrolled vs LQR vs Instantaneous)
    for filename, ylabel, get_unc, get_lqr, get_inst in plot_configs:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        for col, (name, r) in enumerate(results.items()):
            t = r["t"]
            axes[col].plot(t, get_unc(r), "b-", lw=0.8, label="Uncontrolled")
            axes[col].plot(t, get_lqr(r), "r-", lw=0.8, label="LQR")
            axes[col].plot(t, get_inst(r), "g--", lw=0.8, label="Instantaneous")
            axes[col].set_xlabel("Time (s)")
            axes[col].set_title(name)
            axes[col].legend(fontsize=8)
            axes[col].grid(True, alpha=0.3)

        axes[0].set_ylabel(ylabel)
        fig.suptitle(
            ylabel.replace(" (m)", "").replace(" (m/s²)", "").replace(" (MN)", ""),
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(f"{filename}.png", dpi=150)
        plt.close()

    # Plot control forces for BOTH Floor 1 and Floor 8 (Roof)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for col, (name, r) in enumerate(results.items()):
        t = r["t"]

        # Floor 1 Control Force
        axes[0, col].plot(t, r["u_lqr"][:, 0], "r-", lw=0.8, label="LQR")
        axes[0, col].plot(t, r["u_inst"][:, 0], "g--", lw=0.8, label="Instantaneous")
        axes[0, col].set_title(f"{name} - Floor 1")
        axes[0, col].set_ylabel("Control Force (kN)")
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True, alpha=0.3)

        # Floor 8 (Roof) Control Force
        axes[1, col].plot(t, r["u_lqr"][:, -1], "r-", lw=0.8, label="LQR")
        axes[1, col].plot(t, r["u_inst"][:, -1], "g--", lw=0.8, label="Instantaneous")
        axes[1, col].set_title(f"{name} - Floor 8 (Roof)")
        axes[1, col].set_xlabel("Time (s)")
        axes[1, col].set_ylabel("Control Force (kN)")
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(True, alpha=0.3)

    fig.suptitle("Control Forces - Floor 1 and Floor 8", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("control_forces.png", dpi=150)
    plt.close()

    print("\nPlots saved: roof_displacement.png, floor1_displacement.png,")
    print("             roof_acceleration.png, base_shear.png, control_forces.png")


def print_full_table(results):
    """Print comprehensive results table in Markdown format."""
    n = NUM_STORIES

    print("\n## Complete Results Table\n")
    print("| Earthquake | Response | Uncontrolled | LQR | Instantaneous |")
    print("|:-----------|:---------|-------------:|----:|--------------:|")

    for name, r in results.items():
        # Peak values
        d8_unc = np.max(np.abs(r["z_unc"][:, n - 1]))
        d8_lqr = np.max(np.abs(r["z_lqr"][:, n - 1]))
        d8_inst = np.nanmax(np.abs(r["z_inst"][:, n - 1]))

        a8_unc = np.max(np.abs(r["accel_unc"]))
        a8_lqr = np.max(np.abs(r["accel_lqr"]))
        a8_inst = np.nanmax(np.abs(r["accel_inst"]))

        v_unc = np.max(np.abs(r["shear_unc"]))
        v_lqr = np.max(np.abs(r["shear_lqr"]))
        v_inst = np.nanmax(np.abs(r["shear_inst"]))

        f1_lqr = np.max(np.abs(r["u_lqr"][:, 0]))
        f1_inst = np.nanmax(np.abs(r["u_inst"][:, 0]))
        f8_lqr = np.max(np.abs(r["u_lqr"][:, -1]))
        f8_inst = np.nanmax(np.abs(r["u_inst"][:, -1]))

        print(
            f"| **{name}** | Max Roof Disp (m) | {d8_unc:.4f} | {d8_lqr:.4f} | {d8_inst:.4f} |"
        )
        print(
            f"| | Max Roof Accel (m/s²) | {a8_unc:.2f} | {a8_lqr:.2f} | {a8_inst:.2f} |"
        )
        print(f"| | Max Base Shear (kN) | {v_unc:.1f} | {v_lqr:.1f} | {v_inst:.1f} |")
        print(f"| | Max Floor 1 Ctrl Force (kN) | N/A | {f1_lqr:.1f} | {f1_inst:.1f} |")
        print(f"| | Max Floor 8 Ctrl Force (kN) | N/A | {f8_lqr:.1f} | {f8_inst:.1f} |")
        print(
            f"| | Roof Disp Reduction | — | {(1 - d8_lqr / d8_unc) * 100:.1f}% | {(1 - d8_inst / d8_unc) * 100:.1f}% |"
        )


if __name__ == "__main__":
    main()
