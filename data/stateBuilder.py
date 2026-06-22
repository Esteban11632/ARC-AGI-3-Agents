from dataclasses import dataclass
import numpy as np
from perceptron import GridObject, ObjectExtractor

@dataclass
class GameState:
    grid: np.ndarray
    objects: list[GridObject]
    last_action: int | None
    reward: float
    timestep: int

class StateBuilder:
    def __init__(self, perceptron: ObjectExtractor):
        self.perceptron = perceptron
        self.timestep = 0

    def build_state(self, frame: np.ndarray, last_action: int | None, reward: float) -> GameState:
        objects = self.perceptron.extract(frame)
        self.perceptron.track_objects()
        self.timestep += 1
        return GameState(grid=frame, objects=objects, last_action=last_action, reward=reward, timestep=self.timestep)
    
    def get_timestep(self) -> int:
        return self.timestep

"""grid_before = [
    [0, 2, 0, 0],
    [0, 0, 2, 0],
    [0, 0, 0, 3],
    [3, 0, 0, 3],
]

grid_after = [
    [0, 0, 2, 0],
    [0, 0, 2, 0],
    [0, 0, 0, 3],
    [3, 0, 0, 3],
]

state_builder = StateBuilder(ObjectExtractor(grid_height=4, grid_width=4))
state = state_builder.build_state(grid_before, None, 0)
state = state_builder.build_state(grid_after, None, 0)
print(state)"""