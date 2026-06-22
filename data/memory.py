from dataclasses import dataclass, field
from stateBuilder import GameState

@dataclass
class Transition:
    state: GameState
    action: int
    reward: float
    next_state: GameState
    done: bool

@dataclass
class EpisodeMemory:
    transitions: list[Transition] = field(default_factory=list)

    def add(self, state, action, reward, next_state, done=False):
        self.transitions.append(
            Transition(state, action, reward, next_state, done)
        )

    def reset(self):
        self.transitions.clear()

    def last(self):
        return self.transitions[-1] if self.transitions else None

    def get_states(self):
        return [t.state for t in self.transitions]

    def get_actions(self):
        return [t.action for t in self.transitions]

    def get_rewards(self):
        return [t.reward for t in self.transitions]

    def __len__(self):
        return len(self.transitions)