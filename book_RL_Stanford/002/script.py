import itertools
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Process:
    α: float  # Pull strength parameter in the closed interval [0, 1]

    @dataclass
    class State:
        X: int  # Price
        X_1: int  # Previous price

    def up_probablity(self, st: State) -> float:
        if st.X_1 is None:
            return 0.5
        return 0.5 * (1 - self.α * (st.X - st.X_1))

    # Sample from probability distribution
    # True: price will go up
    # False: price will come down
    def is_next_sample_up(self, st: State) -> bool:
        # Binomial: returns integer count of successes
        sample = np.random.binomial(1, self.up_probablity(st))
        return sample == 1

    def next_state(self, st: State) -> State:
        X_1: int
        X: int

        X_1 = st.X
        if self.is_next_sample_up(st):
            X = st.X + 1
        else:
            X = st.X - 1

        new_state = Process.State(X_1=X_1, X=X)
        return new_state


def simulation(ps: Process, start_st: Process.State):
    state = start_st
    while True:
        yield state
        state = ps.next_state(state)


def visualize(prices, label):
    plt.figure(figsize=(10, 6))

    for i in range(prices.shape[0]):
        plt.plot(prices[i], linewidth=1, alpha=0.7, label=f"Simulation {i + 1}")

    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.title(f"price simulation: {label}")
    plt.grid(True, alpha=0.3)

    plt.legend(loc="best", framealpha=0.7, fontsize=9)

    plt.tight_layout()
    plt.show()


prices = np.vstack(
    [
        np.fromiter(
            (
                s.X  # Price
                for s in itertools.islice(
                    simulation(
                        ps=Process(α=0.5), start_st=Process.State(X_1=None, X=0)
                    ),
                    100,  # Time steps
                )
            ),
            int,
        )
        for _ in range(3)  # Number of simulations
    ]
)

print(prices)
visualize(prices=prices, label="")
