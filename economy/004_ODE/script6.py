import matplotlib.pyplot as plt
import numpy as np

# Define the system:
#
# dy1/dt = 6 y1 - y2
# dy2/dt = y1 + 4 y2


# dy1/dt
def f1(y1, y2):
    return 6 * y1 - y2


# dy2/dt
def f2(y1, y2):
    return y1 + 4 * y2


def A():
    A = np.array([[6, -1], [1, 4]])

    eigenvalues, eigenvectors = np.linalg.eig(A)

    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)


def scale_field():
    y1_range = np.linspace(-15, 15, 30)
    y2_range = np.linspace(-15, 15, 30)
    y1, y2 = np.meshgrid(y1_range, y2_range)

    U = f1(y1, y2)
    V = f2(y1, y2)

    magnitude = np.sqrt(U**2 + V**2)
    magnitude[magnitude == 0] = 1
    U_norm = U / magnitude
    V_norm = V / magnitude

    plt.figure(figsize=(9, 8))
    q = plt.quiver(y1, y2, U_norm, V_norm, magnitude, cmap="viridis", alpha=0.8)
    plt.colorbar(q, label="Velocity or Rate Magnitude")
    plt.xlabel("$y_1$")
    plt.ylabel("$y_2$")
    plt.suptitle("Phase Portrait")
    plt.title("$dy_1/dt = 6 * y_1 - y_2$ \n $dy_2/dt = y_1 + 4 * y_2$")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    A()
    scale_field()
