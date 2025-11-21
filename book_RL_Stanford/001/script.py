from dataclasses import dataclass


@dataclass
class process:
    @dataclass
    class state:
        price: float

    def next_state(s: state):
        new_state = state(s.price + 1.0)
        return new_state


def simulation(process, start_state):
    state = start_state
    while True:
        yield state
        state = process.next_state(state)
