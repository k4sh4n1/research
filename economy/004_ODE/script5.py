import matplotlib.pyplot as plt
import numpy as np

# Define the system:
#
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


# dy1/dt = y1 ( a - b y2 )
def f1(y1, y2):
    return y1 * (a - b * y2)


# dy2/dt = c y2 ( y1 - d )
def f2(y1, y2):
    return c * y2 * (y1 - d)


def scale_field():
    y1_range = np.linspace(0, 25, 26)
    y2_range = np.linspace(0, 25, 26)
    y1, y2 = np.meshgrid(y1_range, y2_range)

    U = f1(y1, y2)
    V = f2(y1, y2)

    magnitude = np.sqrt(U**2 + V**2)
    magnitude[magnitude == 0] = 1
    U_norm = U / magnitude
    V_norm = V / magnitude

    plt.figure(figsize=(9, 8))
    q = plt.quiver(y1, y2, U_norm, V_norm, magnitude, cmap="viridis", alpha=0.8)
    plt.colorbar(q, label="Velocity Magnitude")
    plt.xlabel("$y_1$ : Rabbit Population")
    plt.ylabel("$y_2$ : Fox Population")
    plt.suptitle("Phase Portrait: Predator–Prey Systems")
    plt.title("$dy_1/dt = y_1(a - by_2)$ \n $dy_2/dt = cy_2(y_1 - d)$")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    scale_field()
