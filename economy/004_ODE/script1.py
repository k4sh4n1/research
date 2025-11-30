import matplotlib.pyplot as plt
import numpy as np


# Harmonic oscillator: d²y/dt² + y = 0
# Define the system: dy1/dt = y2, dy2/dt = -y1
def f1(y1, y2):
    return y2


def f2(y1, y2):
    return -y1


def scale_field():
    # Create grid of points
    y1_range = np.linspace(-3, 3, 15)  # 15 points from -3 to 3
    y2_range = np.linspace(-3, 3, 15)
    y1, y2 = np.meshgrid(y1_range, y2_range)

    # Compute vector components at each grid point
    U = f1(y1, y2)  # x-component of velocity
    V = f2(y1, y2)  # y-component of velocity

    # Draw the vector field
    plt.figure(figsize=(8, 8))
    plt.quiver(y1, y2, U, V, alpha=0.7)
    plt.xlabel("$y_1$ (position)")
    plt.ylabel("$y_2$ (velocity)")
    plt.suptitle("Phase Portrait: Harmonic Oscillator")
    plt.title("d²y/dt² + y = 0 \n converted to: \n dy1/dt = y2 \n dy2/dt = -y1")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    scale_field()
