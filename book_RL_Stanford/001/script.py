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


def simulation(process, start_state):
    state = start_state
    while True:
        yield state
        state = process.next_state(state)


ps = process()
start_st = process.state(price=0.0)
print(
    np.array(
        np.fromiter(
            (
                s.price
                for s in itertools.islice(
                    simulation(process=ps, start_state=start_st), 100
                )
            ),
            float,
        ),
    )
)
