import numpy as np


def steepest_descent(H, c, x0, tol=1e-8, max_iter=100):
    """
    Steepest Descent Method for Quadratic Functions

    Minimizes: f(x) = (1/2) * x^T * H * x + c^T * x
    Gradient:  g(x) = H * x + c

    Parameters:
    -----------
    H : numpy.ndarray
        Hessian matrix (symmetric positive definite)
    c : numpy.ndarray
        Linear term coefficient vector
    x0 : numpy.ndarray
        Initial starting point
    tol : float
        Convergence tolerance (based on gradient norm)
    max_iter : int
        Maximum number of iterations

    Returns:
    --------
    x : numpy.ndarray
        Optimal solution
    history : list
        History of all iterations
    """

    x = x0.copy().astype(float)
    history = []

    print("=" * 70)
    print("STEEPEST DESCENT OPTIMIZATION")
    print("=" * 70)
    print(f"\nHessian Matrix H:\n{H}")
    print(f"\nLinear term c: {c.flatten()}")
    print(f"\nStarting point x0: {x0.flatten()}")
    print(f"Tolerance: {tol}")
    print("\n" + "=" * 70)

    for k in range(max_iter):
        # Compute gradient: g = H*x + c
        g = H @ x + c
        grad_norm = np.linalg.norm(g)

        # Compute function value: f = 0.5 * x^T * H * x + c^T * x
        f_val = 0.5 * x.T @ H @ x + c.T @ x

        # Store iteration data
        history.append(
            {
                "iteration": k,
                "x": x.copy(),
                "gradient": g.copy(),
                "grad_norm": grad_norm,
                "f_value": float(f_val),
            }
        )

        # Print iteration info
        print(f"\n--- Iteration {k} ---")
        print(f"x = [{x[0, 0]:12.8f}, {x[1, 0]:12.8f}]")
        print(f"g = [{g[0, 0]:12.8f}, {g[1, 0]:12.8f}]")
        print(f"||g|| = {grad_norm:.10f}")
        print(f"f(x) = {float(f_val):.10f}")

        # Check convergence
        if grad_norm < tol:
            print("\n" + "=" * 70)
            print("CONVERGED!")
            print("=" * 70)
            break

        # Compute step size: alpha = (g^T * g) / (g^T * H * g)
        gTg = float(g.T @ g)
        gTHg = float(g.T @ H @ g)
        alpha = gTg / gTHg

        print(f"α = (g^T g) / (g^T H g) = {gTg:.8f} / {gTHg:.8f} = {alpha:.10f}")

        # Update: x_new = x - alpha * g
        x = x - alpha * g

    else:
        print("\n" + "=" * 70)
        print(f"WARNING: Maximum iterations ({max_iter}) reached!")
        print("=" * 70)

    # Final summary
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Optimal x* = [{x[0, 0]:.10f}, {x[1, 0]:.10f}]")
    print(f"Final gradient norm: {grad_norm:.2e}")
    print(f"Total iterations: {len(history)}")
    print(f"Final f(x*) = {float(0.5 * x.T @ H @ x + c.T @ x):.10f}")

    # Compute analytical solution for comparison
    x_analytical = -np.linalg.inv(H) @ c
    print(f"\nAnalytical solution: x* = {x_analytical.flatten()}")

    return x, history


def print_fraction_form(x):
    """Display approximate fractional representation"""
    from fractions import Fraction

    print("\nApproximate Fractional Form:")
    for i, val in enumerate(x.flatten()):
        frac = Fraction(val).limit_denominator(1000)
        print(f"  x{i + 1} ≈ {frac} = {float(frac):.10f}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Define the problem (from your handwritten notes)
    # Hessian matrix
    H = np.array([[2, 1], [1, 2]], dtype=float)

    # Linear term (gradient = Hx + c, so c = [-1, 1]^T)
    c = np.array([[-1], [1]], dtype=float)

    # Starting point
    x0 = np.array([[2], [2]], dtype=float)

    # Run steepest descent
    x_opt, history = steepest_descent(H, c, x0, tol=1e-10, max_iter=50)

    # Show fractional approximation
    print_fraction_form(x_opt)

    # ==========================================================================
    # OPTIONAL: Plot convergence
    # ==========================================================================
    try:
        import matplotlib.pyplot as plt

        # Extract data for plotting
        iterations = [h["iteration"] for h in history]
        grad_norms = [h["grad_norm"] for h in history]
        f_values = [h["f_value"] for h in history]
        x1_vals = [h["x"][0, 0] for h in history]
        x2_vals = [h["x"][1, 0] for h in history]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Gradient norm convergence
        axes[0, 0].semilogy(iterations, grad_norms, "b-o", linewidth=2, markersize=6)
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("||∇f(x)||")
        axes[0, 0].set_title("Gradient Norm Convergence")
        axes[0, 0].grid(True)

        # Plot 2: Function value convergence
        axes[0, 1].plot(iterations, f_values, "r-o", linewidth=2, markersize=6)
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("f(x)")
        axes[0, 1].set_title("Objective Function Value")
        axes[0, 1].grid(True)

        # Plot 3: Trajectory in x-space
        axes[1, 0].plot(x1_vals, x2_vals, "g-o", linewidth=2, markersize=8)
        axes[1, 0].plot(x1_vals[0], x2_vals[0], "ko", markersize=12, label="Start")
        axes[1, 0].plot(x1_vals[-1], x2_vals[-1], "r*", markersize=15, label="End")
        axes[1, 0].set_xlabel("x₁")
        axes[1, 0].set_ylabel("x₂")
        axes[1, 0].set_title("Optimization Trajectory")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].axis("equal")

        # Plot 4: Contour plot with trajectory
        x1_range = np.linspace(-2, 3, 100)
        x2_range = np.linspace(-2, 3, 100)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        Z = (
            0.5 * (H[0, 0] * X1**2 + 2 * H[0, 1] * X1 * X2 + H[1, 1] * X2**2)
            + c[0, 0] * X1
            + c[1, 0] * X2
        )

        axes[1, 1].contour(X1, X2, Z, levels=30, cmap="viridis")
        axes[1, 1].plot(x1_vals, x2_vals, "r-o", linewidth=2, markersize=6)
        axes[1, 1].plot(x1_vals[0], x2_vals[0], "ko", markersize=12, label="Start")
        axes[1, 1].plot(x1_vals[-1], x2_vals[-1], "r*", markersize=15, label="Optimal")
        axes[1, 1].set_xlabel("x₁")
        axes[1, 1].set_ylabel("x₂")
        axes[1, 1].set_title("Contour Plot with Trajectory")
        axes[1, 1].legend()
        axes[1, 1].axis("equal")

        plt.tight_layout()
        plt.savefig("steepest_descent_convergence.png", dpi=150)
        plt.show()

        print("\n✓ Plots saved to 'steepest_descent_convergence.png'")

    except ImportError:
        print("\n(Install matplotlib to see convergence plots)")
