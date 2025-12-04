import matplotlib.pyplot as plt
import numpy as np

# Define the system:
# dx/dt = x(a - by)
# dy/dt = cy(x - d)
#
# or
#
# dy1/dt = y1 ( a - b y2 )
# dy2/dt = c y2 ( y1 - d )

a = 3
b = 1
c = 2
d = 3


# Horizontal component of velocity
def f1(y1, y2):
    return y1 * (a - b * y2)


# Vertical component of velocity
def f2(y1, y2):
    return c * y2 * (y1 - d)


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
    plt.xlabel("$y_1$")
    plt.ylabel("$y_2$")
    plt.suptitle("Phase Portrait")
    plt.title("dy1/dt = y1 ( a - b y2 ) \n dy2/dt = c y2 ( y1 - d )")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    scale_field()
