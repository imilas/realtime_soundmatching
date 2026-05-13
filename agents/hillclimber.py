"""
Hill climber agent - greedy search down the loss landscape.
"""

import numpy as np

from .base import AgentBase


class HillClimberAgent(AgentBase):
    """
    Greedy hill climber: try ±step, accept if loss improves,
    try opposite direction if not, stay put if neither improves.
    """

    def __init__(self, step_percent: float = 5.0):
        super().__init__(step_percent)
        self._last_direction = None

    def step(self, param_range: tuple[float, float]) -> tuple:
        """Try to move downhill on the loss landscape."""
        param_min, param_max = param_range

        if self.current_value is None:
            # First step: return two copies of middle (no better alternative yet)
            middle = (param_min + param_max) / 2.0
            return (middle, middle)

        step_size = (param_max - param_min) * (self.step_percent / 100.0)

        # Try a random direction
        direction = np.random.choice([-1, 1])
        candidate1 = float(np.clip(self.current_value + direction * step_size, param_min, param_max))

        # Try opposite direction as fallback
        candidate2 = float(np.clip(self.current_value - direction * step_size, param_min, param_max))

        return candidate1, candidate2

    def accept_trial(self, trial1_loss: float, trial2_loss: float) -> int:
        """
        Decide which trial to accept (0=stay, 1=trial1, 2=trial2).
        Return the index of the best option.
        """
        if trial1_loss < self.current_loss:
            return 1  # accept trial1
        elif trial2_loss < self.current_loss:
            return 2  # accept trial2
        else:
            return 0  # stay put
