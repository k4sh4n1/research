import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class Process:
    @dataclass
    class State:
        price: float

    @staticmethod
    def next_state(st: State) -> State:
        new_state = Process.State(price=st.price + 1.0)
        return new_state


def simulation(ps: Process, start_st: Process.State):
    state = start_st
    while True:
        yield state
        state = Process.next_state(state)


print(
    np.fromiter(
        (
            s.price
            for s in itertools.islice(
                simulation(ps=Process(), start_st=Process.State(price=0.0)),
                100,
            )
        ),
        float,
    ),
)
