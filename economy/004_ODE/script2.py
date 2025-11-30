import matplotlib.pyplot as plt
import numpy as np

# Damped harmonic oscillator: d²y/dt² + 0.5*dy/dt + y = 0
# Converts to: dx1/dt = x2, dx2/dt = -x1 - 0.5*x2


def f1(x1, x2):
    return x2


def f2(x1, x2):
    return -x1 - 0.5 * x2  # Added damping term


def scale_field():
    # Create grid of points
    x1_range = np.linspace(-3, 3, 15)  # 15 points from -3 to 3
    x2_range = np.linspace(-3, 3, 15)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    # Compute vector components at each grid point
    U = f1(X1, X2)  # x-component of velocity
    V = f2(X1, X2)  # y-component of velocity

    # Draw the vector field
    plt.figure(figsize=(8, 8))
    plt.quiver(X1, X2, U, V, alpha=0.7)
    plt.xlabel("$x_1$ (position)")
    plt.ylabel("$x_2$ (velocity)")
    plt.title("Phase Portrait: Damped Harmonic Oscillator")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    scale_field()
