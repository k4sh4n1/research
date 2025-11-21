import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class process:
    @dataclass
    class State:
        price: float

    @staticmethod
    def next_state(st: State) -> State:
        new_state = process.State(price=st.price + 1.0)
        return new_state


def simulation(ps: process, start_st: process.State):
    state = start_st
    while True:
        yield state
        state = process.next_state(state)


print(
    np.fromiter(
        (
            s.price
            for s in itertools.islice(
                simulation(ps=process(), start_st=process.State(price=0.0)),
                100,
            )
        ),
        float,
    ),
)
