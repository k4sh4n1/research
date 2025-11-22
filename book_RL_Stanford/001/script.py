import itertools
import string
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Process:
    L: int  # Arbitrary reference level for price
    α: float  # Pull strength >= 0

    @dataclass
    class State:
        price: int  # Could be in `pip` units

    def logistic_function(self, st: State) -> float:
        return 1 / (1 + np.exp(-self.α * (self.L - st.price)))

    # Sample from probability distribution
    # True: price will go up
    # False: price will come down
    def is_next_sample_up(self, st: State) -> bool:
        # Binomial: returns integer count of successes
        sample = np.random.binomial(1, self.logistic_function(st), 1)
        return True if sample == 1 else False

    def next_state(self, st: State) -> State:
        new_price: int
        if self.is_next_sample_up(st):
            new_price = st.price + 1
        else:
            new_price = st.price - 1

        new_state = Process.State(price=new_price)
        return new_state


def simulation(ps: Process, start_st: Process.State):
    state = start_st
    while True:
        yield state
        state = ps.next_state(state)


def visualize(prices, label):
    plt.figure(figsize=(10, 6))
    plt.plot(prices, linewidth=2)
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.title(f"price simulation: {label}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


prices = np.fromiter(
    (
        s.price
        for s in itertools.islice(
            simulation(ps=Process(50, 2.0), start_st=Process.State(price=0)),
            100,
        )
    ),
    int,
)

print(prices)
visualize(prices=prices, label="logistic function")
