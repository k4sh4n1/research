"""
Active Structural Control - Series 4
Classic LQR and Instantaneous Optimal Control for 8-story building.
"""

import matplotlib.pyplot as plt
import numpy as np
from building import build_state_space, build_system_matrices, compute_lqr_gain
from config import *
from simulation import simulate_instantaneous, simulate_lqr, simulate_uncontrolled

from earthquake import read_peer_record


def compute_responses(z, M, K, accel_g, dt, g=9.81):
    """Extract roof displacement, roof acceleration, and base shear."""
    n = NUM_STORIES

    # Roof displacement (relative to ground)
    disp_roof = z[:, n - 1]

    # Roof acceleration (absolute) = relative + ground
    accel_rel = z[:, 2 * n - 1]  # roof velocity derivative approximation
    accel_abs = np.gradient(z[:, 2 * n - 1], dt) + accel_g[: len(z)] * g

    # Base shear = K[0,:] @ x + C[0,:] @ v (first story force)
    # Simplified: sum of m * a_abs for all floors
    base_shear = np.zeros(len(z))
    for i in range(len(z)):
        idx = min(i, len(accel_g) - 1)
        a_abs = np.gradient(z[:, n : 2 * n], dt, axis=0)[i] + accel_g[idx] * g
        base_shear[i] = MASS * np.sum(a_abs)

    return disp_roof, accel_abs, base_shear


def find_q_scale(A, Bu, Br, z0, accel_g, dt, target_ratio, tol=0.01):
    """
    Find Q scaling factor to achieve target displacement reduction.

    Uses binary search to find α such that max|x_controlled| = target_ratio * max|x_uncontrolled|
    """
    n = NUM_STORIES
    R = R_MATRIX

    # Get uncontrolled max displacement
    _, z_unc = simulate_uncontrolled(A, Br, z0, accel_g, dt)
    max_unc = np.max(np.abs(z_unc[:, n - 1]))
    target_disp = target_ratio * max_unc

    print(f"  Uncontrolled max roof disp: {max_unc:.4f} m")
    print(f"  Target (25%): {target_disp:.4f} m")

    # Binary search for α
    alpha_low, alpha_high = 1e-6, 1e6

    for iteration in range(50):
        alpha = np.sqrt(alpha_low * alpha_high)
        Q = alpha * np.eye(2 * n)

        try:
            K, _ = compute_lqr_gain(A, Bu, Q, R)
            _, z_ctrl, _ = simulate_lqr(A, Bu, Br, K, z0, accel_g, dt)
            max_ctrl = np.max(np.abs(z_ctrl[:, n - 1]))

            ratio = max_ctrl / max_unc

            if abs(ratio - target_ratio) < tol:
                print(f"  Found α = {alpha:.2e} (ratio = {ratio:.3f})")
                return alpha, K
            elif ratio > target_ratio:
                alpha_low = alpha  # Need more control
            else:
                alpha_high = alpha  # Less control needed
        except:
            alpha_low = alpha

    print(f"  Warning: Converged to α = {alpha:.2e} (ratio = {ratio:.3f})")
    return alpha, K


def main():
    # Build system
    M, C, K = build_system_matrices(NUM_STORIES, MASS, STIFFNESS, DAMPING)
    A, Bu, Br = build_state_space(M, C, K)
    z0 = np.zeros(2 * NUM_STORIES)

    print("=" * 65)
    print("ACTIVE STRUCTURAL CONTROL - 8-STORY BUILDING")
    print("=" * 65)

    results = {}

    for record_name, filename in RECORD_FILES.items():
        print(f"\n{'─' * 65}")
        print(f"Earthquake: {record_name}")
        print("─" * 65)

        dt, accel_g = read_peer_record(filename)
        print(
            f"  Record: {len(accel_g)} pts, dt={dt}s, duration={len(accel_g) * dt:.1f}s"
        )

        # Find Q scaling for 25% target
        print(f"\nTuning Q matrix...")
        alpha, K_lqr = find_q_scale(A, Bu, Br, z0, accel_g, dt, TARGET_REDUCTION)
        Q = alpha * np.eye(2 * NUM_STORIES)

        # Simulate all cases
        t, z_unc = simulate_uncontrolled(A, Br, z0, accel_g, dt)
        t, z_lqr, u_lqr = simulate_lqr(A, Bu, Br, K_lqr, z0, accel_g, dt)
        t, z_inst, u_inst = simulate_instantaneous(
            A, Bu, Br, Q, R_MATRIX, z0, accel_g, dt
        )

        # Store results
        n = NUM_STORIES
        results[record_name] = {
            "t": t,
            "disp_unc": z_unc[:, n - 1],
            "disp_lqr": z_lqr[:, n - 1],
            "disp_inst": z_inst[:, n - 1],
            "disp_1_unc": z_unc[:, 0],
            "disp_1_lqr": z_lqr[:, 0],
            "disp_1_inst": z_inst[:, 0],
            "u_lqr": u_lqr,
            "u_inst": u_inst,
            "alpha": alpha,
        }

        # Print peak responses
        print(f"\n  {'Response':<25} {'Uncontrolled':>12} {'LQR':>12} {'Instant':>12}")
        print(f"  {'-' * 65}")
        print(
            f"  {'Max Roof Disp (m)':<25} {np.max(np.abs(z_unc[:, n - 1])):>12.4f} "
            f"{np.max(np.abs(z_lqr[:, n - 1])):>12.4f} {np.max(np.abs(z_inst[:, n - 1])):>12.4f}"
        )
        print(
            f"  {'Max Floor 1 Disp (m)':<25} {np.max(np.abs(z_unc[:, 0])):>12.4f} "
            f"{np.max(np.abs(z_lqr[:, 0])):>12.4f} {np.max(np.abs(z_inst[:, 0])):>12.4f}"
        )
        print(
            f"  {'Max Control Force (kN)':<25} {'N/A':>12} "
            f"{np.max(np.abs(u_lqr)):>12.1f} {np.max(np.abs(u_inst)):>12.1f}"
        )

    # Plotting
    plot_results(results)
    print_summary_table(results)


def plot_results(results):
    """Generate comparison plots."""
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))

    for col, (name, res) in enumerate(results.items()):
        t = res["t"]

        # Row 0: Roof displacement
        axes[0, col].plot(t, res["disp_unc"], "b-", lw=0.7, label="Uncontrolled")
        axes[0, col].plot(t, res["disp_lqr"], "r-", lw=0.7, label="LQR")
        axes[0, col].plot(t, res["disp_inst"], "g--", lw=0.7, label="Instantaneous")
        axes[0, col].set_ylabel("Roof Disp (m)")
        axes[0, col].set_title(f"{name} (α = {res['alpha']:.2e})")
        axes[0, col].legend(loc="upper right", fontsize=8)
        axes[0, col].grid(True, alpha=0.3)

        # Row 1: Floor 1 displacement
        axes[1, col].plot(t, res["disp_1_unc"], "b-", lw=0.7)
        axes[1, col].plot(t, res["disp_1_lqr"], "r-", lw=0.7)
        axes[1, col].plot(t, res["disp_1_inst"], "g--", lw=0.7)
        axes[1, col].set_ylabel("Floor 1 Disp (m)")
        axes[1, col].grid(True, alpha=0.3)

        # Row 2: Control force at roof (LQR)
        axes[2, col].plot(t, res["u_lqr"][:, -1], "r-", lw=0.7, label="LQR")
        axes[2, col].plot(t, res["u_inst"][:, -1], "g--", lw=0.7, label="Instant")
        axes[2, col].set_ylabel("Roof Force (kN)")
        axes[2, col].legend(loc="upper right", fontsize=8)
        axes[2, col].grid(True, alpha=0.3)

        # Row 3: Control force at floor 1
        axes[3, col].plot(t, res["u_lqr"][:, 0], "r-", lw=0.7)
        axes[3, col].plot(t, res["u_inst"][:, 0], "g--", lw=0.7)
        axes[3, col].set_ylabel("Floor 1 Force (kN)")
        axes[3, col].set_xlabel("Time (s)")
        axes[3, col].grid(True, alpha=0.3)

    plt.suptitle(
        "Active Control: Uncontrolled vs LQR vs Instantaneous Optimal",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("active_control_results.png", dpi=150)
    plt.show()


def print_summary_table(results):
    """Print summary comparison table."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(
        f"{'Earthquake':<12} {'Response':<20} {'Uncontrolled':>12} {'LQR':>12} "
        f"{'Instant':>12} {'Ratio':>10}"
    )
    print("-" * 80)

    for name, res in results.items():
        n = NUM_STORIES
        max_unc = np.max(np.abs(res["disp_unc"]))
        max_lqr = np.max(np.abs(res["disp_lqr"]))
        max_inst = np.max(np.abs(res["disp_inst"]))

        print(
            f"{name:<12} {'Max Roof Disp (m)':<20} {max_unc:>12.4f} {max_lqr:>12.4f} "
            f"{max_inst:>12.4f} {max_lqr / max_unc:>10.2%}"
        )

        max_f_lqr = np.max(np.abs(res["u_lqr"]))
        max_f_inst = np.max(np.abs(res["u_inst"]))
        print(
            f"{'':<12} {'Max Control (kN)':<20} {'N/A':>12} {max_f_lqr:>12.1f} "
            f"{max_f_inst:>12.1f} {'':<10}"
        )
        print("-" * 80)


if __name__ == "__main__":
    main()
