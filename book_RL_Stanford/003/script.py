import itertools
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from numpy._core.numerictypes import unsignedinteger


@dataclass
class Process:
    α: float  # α ∈ R≥0 is pull strength

    @dataclass
    class State:
        U: unsignedinteger  # Count of previuos ups
        D: unsignedinteger  # Count of previous downs

    def up_probablity(self, st: State) -> float:
        if st.U + st.D == 0 or st.D == 0:
            return 0.5
        return 1 / (1 + ((st.U + st.D) / st.D - 1) ** self.α)

    # Sample from probability distribution
    # True: price will go up
    # False: price will come down
    def is_next_sample_up(self, st: State) -> bool:
        # Binomial: returns integer count of successes
        sample = np.random.binomial(1, self.up_probablity(st))
        return sample == 1

    def next_state(self, st: State) -> State:
        D: unsignedinteger
        U: unsignedinteger
        if self.is_next_sample_up(st):
            D = st.D
            U = st.U + 1
        else:
            D = st.D + 1
            U = st.U

        new_state = Process.State(U=U, D=D)
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


# Prevent from running when imported:
if __name__ == "__main__":
    prices = np.vstack(
        [
            np.fromiter(
                (
                    s.U - s.D  # Price: ups minus downs
                    for s in itertools.islice(
                        simulation(ps=Process(α=2.0), start_st=Process.State(U=0, D=0)),
                        100,  # Time steps
                    )
                ),
                int,
            )
            for _ in range(3)  # Number of simulations
        ]
    )

    print(prices)
    visualize(
        prices=prices,
        label="",
    )

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from helper import plot_terminal_distribution

    plot_terminal_distribution(
        alphas=[0.25, 0.75, 1.0, 1.25, 1.75], T=100, traces=1000, start_price=100
    )
