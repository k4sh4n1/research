"""
Active Structural Control - Series 4 (Corrected)
LQR and Instantaneous Optimal Control for 8-story shear building.
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

    State: z = [x1, x2, ..., xn, ẋ1, ẋ2, ..., ẋn]^T

    Returns: A, Bu, Br, M, K, C
    """
    # Mass matrix (diagonal)
    M = m * np.eye(n)

    # Stiffness matrix (tridiagonal)
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = k if i == n - 1 else 2 * k
        if i > 0:
            K[i, i - 1] = -k
        if i < n - 1:
            K[i, i + 1] = -k

    # Damping matrix (same pattern as K, scaled)
    C = (c / k) * K

    # State-space form
    M_inv = np.linalg.inv(M)

    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)  # ẋ = v
    A[n:, :n] = -M_inv @ K  # v̇ = -M⁻¹Kx - M⁻¹Cv + ...
    A[n:, n:] = -M_inv @ C

    # Control input matrix (force at each floor)
    Bu = np.zeros((2 * n, n))
    Bu[n:, :] = M_inv

    # Earthquake input matrix
    Br = np.zeros((2 * n, 1))
    Br[n:, 0] = -np.ones(n)  # All floors experience -ẍg

    return A, Bu, Br, M, K, C


# =============================================================================
# DISCRETIZATION
# =============================================================================
def discretize_system(A, Bu, Br, dt):
    """
    Discretize continuous system using matrix exponential (exact for LTI).

    ż = Az + Bu·u + Br·w  →  z_{k+1} = Ad·z_k + Bd·u_k + Ed·w_k
    """
    n = A.shape[0]

    # Build augmented matrix for exact discretization
    # [A  Bu  Br]      [Ad  Bd  Ed]
    # [0   0   0] * dt → [0   I   0 ]
    # [0   0   0]      [0   0   I ]

    n_u = Bu.shape[1]
    n_w = Br.shape[1]

    M_aug = np.zeros((n + n_u + n_w, n + n_u + n_w))
    M_aug[:n, :n] = A * dt
    M_aug[:n, n : n + n_u] = Bu * dt
    M_aug[:n, n + n_u :] = Br * dt

    exp_M = expm(M_aug)

    Ad = exp_M[:n, :n]
    Bd = exp_M[:n, n : n + n_u]
    Ed = exp_M[:n, n + n_u :]

    return Ad, Bd, Ed


# =============================================================================
# LQR CONTROL
# =============================================================================
def compute_lqr_gain(A, Bu, Q, R):
    """Solve continuous-time Riccati equation and return gain matrix."""
    P = solve_continuous_are(A, Bu, Q, R)
    K = np.linalg.solve(R, Bu.T @ P)
    return K, P


def simulate_lqr(A, Bu, Br, K, z0, accel_g, dt):
    """Simulate closed-loop system with LQR: u = -Kz"""
    n_steps = len(accel_g)
    t = np.arange(n_steps) * dt

    A_cl = A - Bu @ K  # Closed-loop system

    def dzdt(z, ti):
        idx = min(int(ti / dt), n_steps - 1)
        return A_cl @ z + Br.flatten() * accel_g[idx] * G

    z = odeint(dzdt, z0, t)
    u = -z @ K.T

    return t, z, u


# =============================================================================
# INSTANTANEOUS OPTIMAL CONTROL
# =============================================================================
def simulate_instantaneous(Ad, Bd, Ed, Q, R, z0, accel_g, dt):
    """
    Instantaneous optimal control (discrete-time, one-step lookahead).

    At each step, minimize: J = z_{k+1}^T Q z_{k+1} + u_k^T R u_k
    Subject to: z_{k+1} = Ad·z_k + Bd·u_k + Ed·w_k

    Optimal control: u_k = -inv(Bd'QBd + R) @ Bd'Q @ (Ad·z_k + Ed·w_k)
    """
    n_steps = len(accel_g)
    n_states = Ad.shape[0]
    n_control = Bd.shape[1]

    t = np.arange(n_steps) * dt
    z = np.zeros((n_steps, n_states))
    u = np.zeros((n_steps, n_control))
    z[0] = z0

    # Precompute gain matrices
    BQB_R = Bd.T @ Q @ Bd + R
    BQB_R_inv = np.linalg.inv(BQB_R)
    K1 = BQB_R_inv @ Bd.T @ Q @ Ad
    K2 = BQB_R_inv @ Bd.T @ Q @ Ed

    for i in range(n_steps - 1):
        w_k = accel_g[i] * G

        # Instantaneous optimal control
        u[i] = -K1 @ z[i] - K2.flatten() * w_k

        # State update (exact discrete dynamics)
        z[i + 1] = Ad @ z[i] + Bd @ u[i] + Ed.flatten() * w_k

        # Safety check for numerical stability
        if np.any(np.abs(z[i + 1]) > 1e10):
            print(f"  Warning: Instability detected at step {i}")
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
# Q-MATRIX TUNING
# =============================================================================
def tune_q_matrix(A, Bu, Br, Ad, Bd, Ed, z0, accel_g, dt, target_ratio):
    """
    Find Q scaling to achieve target displacement reduction.
    Uses weighted Q focusing on displacements.
    """
    n = NUM_STORIES
    R = np.eye(n)

    # Uncontrolled response
    _, z_unc = simulate_uncontrolled(A, Br, z0, accel_g, dt)
    max_unc = np.max(np.abs(z_unc[:, n - 1]))
    target = target_ratio * max_unc

    print(f"  Uncontrolled max roof disp: {max_unc:.4f} m")
    print(f"  Target ({target_ratio:.0%}): {target:.4f} m")

    # Q matrix structure: weight displacements more than velocities
    def make_Q(alpha):
        Q = np.zeros((2 * n, 2 * n))
        Q[:n, :n] = alpha * np.eye(n)  # Displacement weights
        Q[n:, n:] = alpha * 0.01 * np.eye(n)  # Velocity weights (smaller)
        return Q

    # Binary search
    alpha_low, alpha_high = 1e-2, 1e12
    best_alpha, best_K, best_ratio = alpha_low, None, 1.0

    for _ in range(60):
        alpha = np.sqrt(alpha_low * alpha_high)
        Q = make_Q(alpha)

        try:
            K, _ = compute_lqr_gain(A, Bu, Q, R)
            _, z_ctrl, _ = simulate_lqr(A, Bu, Br, K, z0, accel_g, dt)
            max_ctrl = np.max(np.abs(z_ctrl[:, n - 1]))
            ratio = max_ctrl / max_unc

            if abs(ratio - target_ratio) < 0.005:
                print(f"  Found α = {alpha:.2e} → ratio = {ratio:.1%}")
                return alpha, K, Q
            elif ratio > target_ratio:
                alpha_low = alpha
            else:
                alpha_high = alpha

            if abs(ratio - target_ratio) < abs(best_ratio - target_ratio):
                best_alpha, best_K, best_ratio = alpha, K, ratio

        except Exception as e:
            alpha_high = alpha

    print(f"  Best: α = {best_alpha:.2e} → ratio = {best_ratio:.1%}")
    return best_alpha, best_K, make_Q(best_alpha)


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Build system
    A, Bu, Br, M, K_mat, C = build_state_space(NUM_STORIES, MASS, STIFFNESS, DAMPING)
    z0 = np.zeros(2 * NUM_STORIES)
    n = NUM_STORIES

    print("=" * 70)
    print("ACTIVE STRUCTURAL CONTROL - 8-STORY BUILDING (CORRECTED)")
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

        # Discretize for instantaneous control
        Ad, Bd, Ed = discretize_system(A, Bu, Br, dt)

        # Tune Q
        print("\nTuning Q matrix...")
        alpha, K_lqr, Q = tune_q_matrix(
            A, Bu, Br, Ad, Bd, Ed, z0, accel_g, dt, TARGET_REDUCTION
        )

        # Simulate
        t, z_unc = simulate_uncontrolled(A, Br, z0, accel_g, dt)
        t, z_lqr, u_lqr = simulate_lqr(A, Bu, Br, K_lqr, z0, accel_g, dt)
        t, z_inst, u_inst = simulate_instantaneous(
            Ad, Bd, Ed, Q, np.eye(n), z0, accel_g, dt
        )

        # Store
        results[rec_name] = {
            "t": t,
            "dt": dt,
            "alpha": alpha,
            "z_unc": z_unc,
            "z_lqr": z_lqr,
            "z_inst": z_inst,
            "u_lqr": u_lqr,
            "u_inst": u_inst,
        }

        # Print summary
        print(f"\n  {'Response':<25} {'Uncontrol':>12} {'LQR':>12} {'Instant':>12}")
        print(f"  {'-' * 55}")

        d_unc = np.max(np.abs(z_unc[:, n - 1]))
        d_lqr = np.max(np.abs(z_lqr[:, n - 1]))
        d_inst = np.nanmax(np.abs(z_inst[:, n - 1]))

        print(
            f"  {'Max Roof Disp (m)':<25} {d_unc:>12.4f} {d_lqr:>12.4f} {d_inst:>12.4f}"
        )
        print(
            f"  {'Ratio':<25} {'100%':>12} {d_lqr / d_unc:>11.1%} {d_inst / d_unc:>11.1%}"
        )
        print(
            f"  {'Max Ctrl Force (kN)':<25} {'N/A':>12} {np.max(np.abs(u_lqr)):>12.1f} {np.nanmax(np.abs(u_inst)):>12.1f}"
        )

    # Plot
    plot_results(results)
    print_table(results)


def plot_results(results):
    """Generate time history plots."""
    n = NUM_STORIES
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))

    for col, (name, r) in enumerate(results.items()):
        t = r["t"]

        # Roof displacement
        axes[0, col].plot(t, r["z_unc"][:, n - 1], "b-", lw=0.7, label="Uncontrolled")
        axes[0, col].plot(t, r["z_lqr"][:, n - 1], "r-", lw=0.7, label="LQR")
        axes[0, col].plot(
            t, r["z_inst"][:, n - 1], "g--", lw=0.7, label="Instantaneous"
        )
        axes[0, col].set_ylabel("Roof Disp (m)")
        axes[0, col].set_title(f"{name}")
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True, alpha=0.3)

        # Floor 1 displacement
        axes[1, col].plot(t, r["z_unc"][:, 0], "b-", lw=0.7)
        axes[1, col].plot(t, r["z_lqr"][:, 0], "r-", lw=0.7)
        axes[1, col].plot(t, r["z_inst"][:, 0], "g--", lw=0.7)
        axes[1, col].set_ylabel("Floor 1 Disp (m)")
        axes[1, col].grid(True, alpha=0.3)

        # Roof control force
        axes[2, col].plot(t, r["u_lqr"][:, -1], "r-", lw=0.7, label="LQR")
        axes[2, col].plot(t, r["u_inst"][:, -1], "g--", lw=0.7, label="Instantaneous")
        axes[2, col].set_ylabel("Roof Force (kN)")
        axes[2, col].legend(fontsize=8)
        axes[2, col].grid(True, alpha=0.3)

        # Floor 1 control force
        axes[3, col].plot(t, r["u_lqr"][:, 0], "r-", lw=0.7)
        axes[3, col].plot(t, r["u_inst"][:, 0], "g--", lw=0.7)
        axes[3, col].set_ylabel("Floor 1 Force (kN)")
        axes[3, col].set_xlabel("Time (s)")
        axes[3, col].grid(True, alpha=0.3)

    plt.suptitle(
        "Active Control Comparison: Uncontrolled vs LQR vs Instantaneous",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("active_control_results.png", dpi=150)
    plt.show()


def print_table(results):
    """Print summary comparison table."""
    n = NUM_STORIES
    print("\n" + "=" * 85)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 85)
    print(
        f"{'Earthquake':<12} {'Metric':<22} {'Uncontrolled':>14} {'LQR':>14} {'Instant':>14}"
    )
    print("-" * 85)

    for name, r in results.items():
        d_unc = np.max(np.abs(r["z_unc"][:, n - 1]))
        d_lqr = np.max(np.abs(r["z_lqr"][:, n - 1]))
        d_inst = np.nanmax(np.abs(r["z_inst"][:, n - 1]))

        d1_unc = np.max(np.abs(r["z_unc"][:, 0]))
        d1_lqr = np.max(np.abs(r["z_lqr"][:, 0]))
        d1_inst = np.nanmax(np.abs(r["z_inst"][:, 0]))

        f_lqr = np.max(np.abs(r["u_lqr"]))
        f_inst = np.nanmax(np.abs(r["u_inst"]))

        print(
            f"{name:<12} {'Max Roof Disp (m)':<22} {d_unc:>14.4f} {d_lqr:>14.4f} {d_inst:>14.4f}"
        )
        print(
            f"{'':<12} {'Max Floor 1 Disp (m)':<22} {d1_unc:>14.4f} {d1_lqr:>14.4f} {d1_inst:>14.4f}"
        )
        print(
            f"{'':<12} {'Max Control (kN)':<22} {'N/A':>14} {f_lqr:>14.1f} {f_inst:>14.1f}"
        )
        print(
            f"{'':<12} {'Roof Disp Ratio':<22} {'100%':>14} {d_lqr / d_unc:>13.1%} {d_inst / d_unc:>13.1%}"
        )
        print("-" * 85)


if __name__ == "__main__":
    main()
