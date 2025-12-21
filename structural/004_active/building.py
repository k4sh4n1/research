"""Build state-space model for shear building."""

import numpy as np
from scipy.linalg import solve_continuous_are


def build_system_matrices(n, m, k, c):
    """
    Build M, C, K matrices for n-story shear building.

    Returns: M, C, K (all n×n)
    """
    M = m * np.eye(n)

    # Tridiagonal stiffness matrix
    K = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            K[i, i] = 2 * k if n > 1 else k
            if n > 1:
                K[i, i + 1] = -k
        elif i == n - 1:
            K[i, i] = k
            K[i, i - 1] = -k
        else:
            K[i, i] = 2 * k
            K[i, i - 1] = -k
            K[i, i + 1] = -k

    # Same pattern for damping
    C = (c / k) * K

    return M, C, K


def build_state_space(M, C, K):
    """
    Convert M, C, K to state-space form: ż = Az + Bu·u + Br·ẍg

    State vector z = [x; ẋ] (displacements, then velocities)

    Returns: A, Bu, Br
    """
    n = M.shape[0]
    M_inv = np.linalg.inv(M)

    # System matrix A (2n × 2n)
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -M_inv @ K
    A[n:, n:] = -M_inv @ C

    # Control input matrix Bu (2n × n) - forces at all floors
    Bu = np.zeros((2 * n, n))
    Bu[n:, :] = M_inv

    # Earthquake input matrix Br (2n × 1)
    Br = np.zeros((2 * n, 1))
    Br[n:, 0] = -np.ones(n)

    return A, Bu, Br


def compute_lqr_gain(A, Bu, Q, R):
    """
    Solve Riccati equation and compute LQR gain matrix.

    Returns: K (gain matrix), P (Riccati solution)
    """
    P = solve_continuous_are(A, Bu, Q, R)
    K = np.linalg.solve(R, Bu.T @ P)
    return K, P
