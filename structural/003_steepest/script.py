from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

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

    Args:
        H: Symmetric positive definite Hessian matrix
        c: Linear term coefficient vector
        x0: Initial starting point
        tol: Convergence tolerance (gradient norm)
        max_iter: Maximum iterations
        verbose: Print iteration details

    Returns:
        Optimal solution and iteration history
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


def plot_convergence(
    history: list[IterationData], H: np.ndarray, c: np.ndarray
) -> None:
    """Generate convergence visualization plots."""
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
    Z = (
        0.5 * (H[0, 0] * X1**2 + 2 * H[0, 1] * X1 * X2 + H[1, 1] * X2**2)
        + c[0, 0] * X1
        + c[1, 0] * X2
    )

    axes[1, 1].contour(X1, X2, Z, levels=30, cmap="viridis")
    axes[1, 1].plot(x1, x2, "r-o", lw=2, ms=5)
    axes[1, 1].plot(x1[0], x2[0], "ko", ms=10, label="Start")
    axes[1, 1].plot(x1[-1], x2[-1], "r*", ms=14, label="Optimal")
    axes[1, 1].set(xlabel="$x_1$", ylabel="$x_2$", title="Contour + Trajectory")
    axes[1, 1].legend()
    axes[1, 1].axis("equal")

    plt.tight_layout()
    plt.savefig("steepest_descent_convergence.png", dpi=150)
    plt.show()
    print("✓ Plot saved to 'steepest_descent_convergence.png'")


if __name__ == "__main__":
    # Problem definition
    H = np.array([[2, 1], [1, 2]], dtype=float)
    c = np.array([[-1], [1]], dtype=float)
    x0 = np.array([[2], [2]], dtype=float)

    # Run optimization
    x_opt, history = steepest_descent(H, c, x0, tol=1e-10, max_iter=50)

    # Visualize
    plot_convergence(history, H, c)
