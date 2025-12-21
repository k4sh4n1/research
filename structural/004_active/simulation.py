"""Time history simulation for controlled and uncontrolled systems."""

import numpy as np
from scipy.integrate import odeint


def simulate_uncontrolled(A, Br, z0, accel_g, dt, g=9.81):
    """
    Simulate uncontrolled system response.

    Returns: t, z (time array, state history)
    """
    n_steps = len(accel_g)
    t = np.arange(n_steps) * dt

    def dzdt(z, t_val):
        idx = min(int(t_val / dt), n_steps - 1)
        return A @ z + Br.flatten() * accel_g[idx] * g

    z = odeint(dzdt, z0, t)
    return t, z


def simulate_lqr(A, Bu, Br, K, z0, accel_g, dt, g=9.81):
    """
    Simulate system with LQR control: u = -Kz

    Returns: t, z, u (time, states, control forces)
    """
    n_steps = len(accel_g)
    t = np.arange(n_steps) * dt
    n_floors = Bu.shape[1]

    A_cl = A - Bu @ K  # Closed-loop system matrix

    def dzdt(z, t_val):
        idx = min(int(t_val / dt), n_steps - 1)
        return A_cl @ z + Br.flatten() * accel_g[idx] * g

    z = odeint(dzdt, z0, t)
    u = -z @ K.T  # Control history

    return t, z, u


def simulate_instantaneous(A, Bu, Br, Q, R, z0, accel_g, dt, g=9.81):
    """
    Simulate system with instantaneous optimal control.

    At each step, minimize: J = z_{k+1}^T Q z_{k+1} + u_k^T R u_k

    Returns: t, z, u
    """
    n_steps = len(accel_g)
    n_states = A.shape[0]
    n_floors = Bu.shape[1]

    t = np.arange(n_steps) * dt
    z = np.zeros((n_steps, n_states))
    u = np.zeros((n_steps, n_floors))
    z[0] = z0

    # Discretize system (simple Euler for clarity)
    Ad = np.eye(n_states) + A * dt
    Bd = Bu * dt

    # Precompute gain for instantaneous control
    # Optimal u = -(R + Bd^T Q Bd)^{-1} Bd^T Q Ad z
    BQB = Bd.T @ Q @ Bd + R
    BQA = Bd.T @ Q @ Ad
    K_inst = np.linalg.solve(BQB, BQA)

    for i in range(n_steps - 1):
        # Instantaneous optimal control
        u[i] = -K_inst @ z[i]

        # State update (Euler integration)
        ag = accel_g[i] * g
        z[i + 1] = z[i] + (A @ z[i] + Bu @ u[i] + Br.flatten() * ag) * dt

    return t, z, u
