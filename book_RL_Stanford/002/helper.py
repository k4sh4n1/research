import itertools

import matplotlib.pyplot as plt
import numpy as np
from script import Process, simulation


def simulate_terminal_price(α: float, T: int, start_price: int = 100) -> int:
    """Run one simulation and return terminal price."""
    prices = [
        s.X  # Price
        for s in itertools.islice(
            simulation(ps=Process(α=α), start_st=Process.State(X=start_price, δ=None)),
            T,  # Time steps
        )
    ]
    return prices[-1]


def get_terminal_prices(
    α: float, T: int, traces: int, start_price: int = 100
) -> np.ndarray:
    """Run multiple simulations and collect terminal prices."""
    return np.array([simulate_terminal_price(α, T, start_price) for _ in range(traces)])


def plot_terminal_distribution(
    alphas: list[float], T: int = 100, traces: int = 1000, start_price: int = 100
):
    """Plot terminal price distribution for multiple α values."""
    styles = [
        {"color": "red", "linestyle": "-", "linewidth": 1.5},
        {"color": "blue", "linestyle": "--", "linewidth": 1.5},
        {"color": "green", "linestyle": "-.", "linewidth": 1.5},
        {"color": "brown", "linestyle": "-", "linewidth": 1.5},
        {"color": "purple", "linestyle": "--", "linewidth": 1.5},
    ]

    plt.figure(figsize=(10, 6))

    for i, α in enumerate(alphas):
        prices = get_terminal_prices(α, T, traces, start_price=start_price)

        # Count occurrences of each price
        unique, counts = np.unique(prices, return_counts=True)

        plt.plot(
            unique,
            counts,
            label=rf"$\alpha_{i + 1} = {α}$",
            **styles[i % len(styles)],
        )

    plt.xlabel("Terminal Stock Price")
    plt.ylabel("Counts")
    plt.title(
        f"Terminal Price Counts (Startprice={start_price}, Timesteps={T}, Simulations={traces})"
    )
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()
