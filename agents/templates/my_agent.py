import random
from typing import Any

import torch
from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from models.prototype import Prototype

class MyAgent(Agent):
    """An agent that always selects actions at random."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Prototype().to(self.device)
        self.model.eval()

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose which action the Agent should take, fill in any arguments, and return it."""
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            # if game is not started (at init or after GAME_OVER) we need to reset
            # add a small delay before resetting after GAME_OVER to avoid timeout
            action = GameAction.RESET
        else:
            # 1. Extract grid
            grid = latest_frame.frame[-1]                    # (64, 64) Python lists
            # 2. To tensor
            x = torch.tensor(grid, dtype=torch.long)        # (64, 64)
            x = x.unsqueeze(0).to(self.device)              # (1, 64, 64)
            # 3. Model forward
            logits = self.model(x)                          # (1, 8) action logits
            # 4. Mask illegal actions
            mask = torch.full((8,), float('-inf'))
            for action_id in latest_frame.available_actions:
                mask[action_id] = 0.0
            logits = logits[0] + mask
            # 5. Pick action
            action_id = logits.argmax().item()              # e.g. 3 = ACTION3
            action = GameAction.from_id(action_id)

        if action.is_simple():
            action.reasoning = f"RNG told me to pick {action.value}"
        elif action.is_complex():
            action.set_data(
                {
                    "x": random.randint(0, 63),
                    "y": random.randint(0, 63),
                }
            )
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": "RNG said so!",
            }
        return action