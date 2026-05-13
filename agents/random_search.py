"""
Random search agent - pure random exploration.
"""

import numpy as np

from .base import AgentBase


class RandomSearchAgent(AgentBase):
    """Random search: sample uniformly from parameter space each step."""

    def step(self, param_range: tuple[float, float]) -> float:
        """Return a random parameter value from the range."""
        param_min, param_max = param_range
        return float(np.random.uniform(param_min, param_max))
