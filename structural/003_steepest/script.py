from dataclasses import dataclass
from fractions import Fraction

import numpy as np


@dataclass
class IterationData:
    """Stores data for a single iteration."""

    iteration: int
    x: np.ndarray
    gradient: np.ndarray
    grad_norm: float
    f_value: float


def steepest_descent(
    H: np.ndarray,
    c: np.ndarray,
    x0: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 100,
    verbose: bool = True,
) -> tuple[np.ndarray, list[IterationData]]:
    """
    Steepest Descent for quadratic functions: f(x) = 0.5 * x'Hx + c'x
    """
    x = x0.astype(float).copy()
    history = []

    if verbose:
        print(f"{'=' * 60}\nSTEEPEST DESCENT\n{'=' * 60}")
        print(f"H:\n{H}\nc: {c.flatten()}\nx0: {x0.flatten()}\n{'=' * 60}")

    for k in range(max_iter):
        g = H @ x + c
        grad_norm = np.linalg.norm(g)
        f_val = float(0.5 * x.T @ H @ x + c.T @ x)

        history.append(IterationData(k, x.copy(), g.copy(), grad_norm, f_val))

        if verbose:
            print(
                f"Iter {k}: x=[{x[0, 0]:.8f}, {x[1, 0]:.8f}], "
                f"||g||={grad_norm:.2e}, f={f_val:.8f}"
            )

        if grad_norm < tol:
            if verbose:
                print(f"{'=' * 60}\nCONVERGED at iteration {k}\n{'=' * 60}")
            break

        alpha = float(g.T @ g) / float(g.T @ H @ g)
        x = x - alpha * g
    else:
        if verbose:
            print(f"WARNING: Max iterations ({max_iter}) reached!")

    if verbose:
        x_analytical = -np.linalg.solve(H, c)
        print(f"\nResult: x*=[{x[0, 0]:.10f}, {x[1, 0]:.10f}]")
        print(f"Analytical: {x_analytical.flatten()}")
        print_fractional(x)

    return x, history


def print_fractional(x: np.ndarray) -> None:
    """Display approximate fractional representation of solution."""
    print("\nFractional approximation:")
    for i, val in enumerate(x.flatten()):
        frac = Fraction(val).limit_denominator(1000)
        print(f"  x{i + 1} ≈ {frac} = {float(frac):.10f}")


def quadratic_function(
    X1: np.ndarray, X2: np.ndarray, H: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Evaluate f(x) = 0.5 * x'Hx + c'x over a meshgrid."""
    return (
        0.5 * (H[0, 0] * X1**2 + 2 * H[0, 1] * X1 * X2 + H[1, 1] * X2**2)
        + c[0, 0] * X1
        + c[1, 0] * X2
    )


def plot_convergence(
    history: list[IterationData], H: np.ndarray, c: np.ndarray
) -> None:
    """Generate 2D convergence visualization plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib for plots: pip install matplotlib")
        return

    iters = [h.iteration for h in history]
    norms = [h.grad_norm for h in history]
    fvals = [h.f_value for h in history]
    x1 = [h.x[0, 0] for h in history]
    x2 = [h.x[1, 0] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Gradient norm (log scale)
    axes[0, 0].semilogy(iters, norms, "b-o", lw=2, ms=5)
    axes[0, 0].set(xlabel="Iteration", ylabel="||∇f(x)||", title="Gradient Norm")
    axes[0, 0].grid(True)

    # Function value
    axes[0, 1].plot(iters, fvals, "r-o", lw=2, ms=5)
    axes[0, 1].set(xlabel="Iteration", ylabel="f(x)", title="Objective Value")
    axes[0, 1].grid(True)

    # Trajectory
    axes[1, 0].plot(x1, x2, "g-o", lw=2, ms=6)
    axes[1, 0].plot(x1[0], x2[0], "ko", ms=10, label="Start")
    axes[1, 0].plot(x1[-1], x2[-1], "r*", ms=14, label="End")
    axes[1, 0].set(xlabel="$x_1$", ylabel="$x_2$", title="Trajectory")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].axis("equal")

    # Contour with trajectory
    r = np.linspace(-2, 3, 100)
    X1, X2 = np.meshgrid(r, r)
    Z = quadratic_function(X1, X2, H, c)

    axes[1, 1].contour(X1, X2, Z, levels=30, cmap="viridis")
    axes[1, 1].plot(x1, x2, "r-o", lw=2, ms=5)
    axes[1, 1].plot(x1[0], x2[0], "ko", ms=10, label="Start")
    axes[1, 1].plot(x1[-1], x2[-1], "r*", ms=14, label="Optimal")
    axes[1, 1].set(xlabel="$x_1$", ylabel="$x_2$", title="Contour + Trajectory")
    axes[1, 1].legend()
    axes[1, 1].axis("equal")

    plt.tight_layout()
    plt.savefig("steepest_descent_convergence.png", dpi=150)
    print("✓ 2D plots saved to 'steepest_descent_convergence.png'")


def plot_3d_surface(history: list[IterationData], H: np.ndarray, c: np.ndarray) -> None:
    """Generate interactive 3D surface plot with optimization trajectory."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib for plots: pip install matplotlib")
        return

    x1 = np.array([h.x[0, 0] for h in history])
    x2 = np.array([h.x[1, 0] for h in history])
    fvals = np.array([h.f_value for h in history])

    r = np.linspace(-2, 3, 80)
    X1, X2 = np.meshgrid(r, r)
    Z = quadratic_function(X1, X2, H, c)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X1, X2, Z, cmap="viridis", alpha=0.7, edgecolor="none")
    ax.plot(x1, x2, fvals, "r-o", lw=3, ms=6, label="Trajectory", zorder=10)
    ax.scatter(
        x1[0], x2[0], fvals[0], c="black", s=100, marker="o", label="Start", zorder=11
    )
    ax.scatter(
        x1[-1],
        x2[-1],
        fvals[-1],
        c="red",
        s=200,
        marker="*",
        label="Optimal",
        zorder=11,
    )

    ax.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$f(x)$")
    ax.set_title(
        "3D Surface with Optimization Trajectory\n(drag to rotate, scroll to zoom)"
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig("steepest_descent_3d.png", dpi=150)
    print("✓ 3D plot saved to 'steepest_descent_3d.png'")


def plot_path_profile(history: list[IterationData]) -> None:
    """Plot function value vs cumulative distance along trajectory (elevation profile)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib for plots: pip install matplotlib")
        return

    # Extract positions and function values
    positions = np.array([h.x.flatten() for h in history])
    fvals = np.array([h.f_value for h in history])

    # Compute cumulative path length
    steps = np.diff(positions, axis=0)
    distances = np.linalg.norm(steps, axis=1)
    cumulative_dist = np.concatenate([[0], np.cumsum(distances)])

    # Create plot
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(cumulative_dist, fvals, "b-o", lw=2, ms=7)
    ax.fill_between(cumulative_dist, fvals, alpha=0.3)
    ax.scatter(cumulative_dist[0], fvals[0], c="black", s=120, zorder=5, label="Start")
    ax.scatter(
        cumulative_dist[-1],
        fvals[-1],
        c="red",
        s=180,
        marker="*",
        zorder=5,
        label="Optimal",
    )

    ax.set(
        xlabel="Cumulative Path Distance",
        ylabel="$f(x)$",
        title="Function Value Along Trajectory (Elevation Profile)",
    )
    ax.legend()
    ax.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig("steepest_descent_path_profile.png", dpi=150)
    print("✓ Path profile saved to 'steepest_descent_path_profile.png'")


def plot_path_profile_detailed(
    history: list[IterationData], H: np.ndarray, c: np.ndarray
) -> None:
    """Plot function value with interpolated points showing true quadratic behavior."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib for plots: pip install matplotlib")
        return

    # Build detailed path with intermediate samples
    all_positions = []
    all_fvals = []

    for i in range(len(history) - 1):
        x_start = history[i].x.flatten()
        x_end = history[i + 1].x.flatten()

        # Sample 50 points along each segment
        for t in np.linspace(0, 1, 50, endpoint=(i == len(history) - 2)):
            x_interp = (1 - t) * x_start + t * x_end
            x_col = x_interp.reshape(-1, 1)
            f_val = float(0.5 * x_col.T @ H @ x_col + c.T @ x_col)
            all_positions.append(x_interp)
            all_fvals.append(f_val)

    all_positions = np.array(all_positions)
    all_fvals = np.array(all_fvals)

    # Compute cumulative distance
    steps = np.diff(all_positions, axis=0)
    distances = np.linalg.norm(steps, axis=1)
    cumulative_dist = np.concatenate([[0], np.cumsum(distances)])

    # Also get iteration points for markers
    iter_positions = np.array([h.x.flatten() for h in history])
    iter_fvals = np.array([h.f_value for h in history])
    iter_steps = np.diff(iter_positions, axis=0)
    iter_distances = np.linalg.norm(iter_steps, axis=1)
    iter_cumulative = np.concatenate([[0], np.cumsum(iter_distances)])

    # Create plot
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(cumulative_dist, all_fvals, "b-", lw=2, label="True path (quadratic)")
    ax.scatter(
        iter_cumulative, iter_fvals, c="blue", s=60, zorder=5, label="Iteration points"
    )
    ax.fill_between(cumulative_dist, all_fvals, alpha=0.3)
    ax.scatter(
        iter_cumulative[0], iter_fvals[0], c="black", s=120, zorder=6, label="Start"
    )
    ax.scatter(
        iter_cumulative[-1],
        iter_fvals[-1],
        c="red",
        s=180,
        marker="*",
        zorder=6,
        label="Optimal",
    )

    ax.set(
        xlabel="Cumulative Path Distance",
        ylabel="$f(x)$",
        title="Function Value Along Trajectory (True Quadratic Shape)",
    )
    ax.legend()
    ax.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig("steepest_descent_path_profile_detailed.png", dpi=150)
    print(
        "✓ Detailed path profile saved to 'steepest_descent_path_profile_detailed.png'"
    )


if __name__ == "__main__":
    # Problem definition
    H = np.array([[2, 1], [1, 2]], dtype=float)
    c = np.array([[-1], [1]], dtype=float)
    x0 = np.array([[2], [2]], dtype=float)

    # Run optimization
    x_opt, history = steepest_descent(H, c, x0, tol=1e-10, max_iter=50)

    # Visualize
    plot_convergence(history, H, c)
    plot_3d_surface(history, H, c)
    plot_path_profile(history)
    plot_path_profile_detailed(history, H, c)

    # Show all plots
    import matplotlib.pyplot as plt

    plt.show()
