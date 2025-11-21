import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class process:
    @dataclass
    class state:
        price: float

    @staticmethod
    def next_state(st: "process.state") -> "process.state":
        new_state = process.state(st.price + 1.0)
        return new_state


def simulation(ps: process, start_st: process.state):
    state = start_st
    while True:
        yield state
        state = process.next_state(state)


print(
    np.array(
        np.fromiter(
            (
                s.price
                for s in itertools.islice(
                    simulation(ps=process(), start_st=process.state(price=0.0)),
                    100,
                )
            ),
            float,
        ),
    )
)
