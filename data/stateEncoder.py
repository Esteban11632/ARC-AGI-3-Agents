import torch
from stateBuilder import GameState
from perceptron import ObjectExtractor
from stateBuilder import StateBuilder

class StateEncoder:
    def __init__(self, grid_height: int, grid_width: int):
        self.grid_height = grid_height
        self.grid_width = grid_width

    def encode(self, state: GameState):

        return {
            "objects": self.encode_objects(state),

            "last_action": torch.tensor(
                [state.last_action if state.last_action is not None else -1],
                dtype=torch.long
            ),

            "reward": torch.tensor(
                [state.reward],
                dtype=torch.float32
            ),

            "timestep": torch.tensor(
                [state.timestep],
                dtype=torch.long
            )
        }

    def encode_objects(self, state: GameState) -> torch.Tensor:
        objects = state.objects
        features = []

        for obj in objects:
            min_row, min_col, max_row, max_col = obj.bbox

            features.append([
                            obj.color,

                            # Normalized center (0 to 1)
                            obj.center[0] / self.grid_height,
                            obj.center[1] / self.grid_width,

                            # Normalized bounding box (0 to 1)
                            min_row / self.grid_height,
                            min_col / self.grid_width,
                            max_row / self.grid_height,
                            max_col / self.grid_width,

                            # Normalized width and height (0 to 1)
                            obj.width / self.grid_width,
                            obj.height / self.grid_height,

                            # Normalized area (0 to 1)
                            obj.area / (self.grid_height * self.grid_width),

                            # Fraction of bbox occupied by object
                            obj.fill_ratio,

                            float(obj.shape_type.value),
                            float(obj.area_ratio),
                            float(obj.is_large_region),
                            obj.delta_x / self.grid_height,
                            obj.delta_y / self.grid_width,
                            float(obj.color_changed),
                            obj.rotation_delta / 360.0,
                            float(obj.rotation_changed),
                            float(obj.position_changed),
                        ])

        features = torch.tensor(features, dtype=torch.float32)

        return features

grid_before = [
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

state_encoder = StateEncoder(grid_height=4, grid_width=4)
print(state)
print(state_encoder.encode_objects(state))