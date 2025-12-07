import matplotlib.pyplot as plt
import numpy as np

# Define the system:
#
# dy/dt = -0.9 * y - 30 * r + 900
# dr/dt = 2 * y - 3 * r + 50


# dy/dt
def f1(y, r):
    return -0.9 * y - 30 * r + 900


# dr/dt
def f2(y, r):
    return 2 * y - 3 * r + 50


def A():
    A = np.array([[-0.9, -30], [2, -3]])

    eigenvalues, eigenvectors = np.linalg.eig(A)

    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)

    # Verify that eigenvectors are already normalized
    for i in range(eigenvectors.shape[1]):
        norm = np.linalg.norm(eigenvectors[:, i])
        print(f"Norm of eigenvector {i + 1}: {norm}")


def scale_field():
    y_range = np.linspace(0, 60, 30)
    r_range = np.linspace(0, 60, 30)
    y, r = np.meshgrid(y_range, r_range)

    U = f1(y, r)
    V = f2(y, r)

    magnitude = np.sqrt(U**2 + V**2)
    magnitude[magnitude == 0] = 1
    U_norm = U / magnitude
    V_norm = V / magnitude

    plt.figure(figsize=(9, 8))
    q = plt.quiver(y, r, U_norm, V_norm, magnitude, cmap="viridis", alpha=0.8)
    plt.colorbar(q, label="Rate Magnitude")
    plt.xlabel("$y$")
    plt.ylabel("$r$")
    plt.suptitle("Phase Portrait")
    plt.title("$dy/dt = -0.9 * y - 30 * r + 900$ \n $dr/dt = 2 * y - 3 * r + 50$")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    A()
    scale_field()
