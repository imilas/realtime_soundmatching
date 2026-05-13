"""
Base classes for agent implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class AgentObservation:
    """What the agent observes each step."""
    iteration: int
    current_value: float
    current_loss: float
    best_value: float
    best_loss: float
    history_values: list[float]
    history_losses: list[float]


class AgentBase(ABC):
    """Abstract base class for optimization agents."""

    def __init__(self, step_percent: float = 5.0):
        """
        Parameters
        ----------
        step_percent : float
            Step size as percentage of parameter range
        """
        self.step_percent = step_percent
        self.iteration = 0
        self.history_values = []
        self.history_losses = []
        self.best_value = None
        self.best_loss = float("inf")
        self.current_value = None
        self.current_loss = float("inf")

    @abstractmethod
    def step(self, param_range: tuple[float, float]) -> float:
        """
        Decide next parameter value to try.

        Parameters
        ----------
        param_range : tuple[float, float]
            (min, max) of parameter bounds

        Returns
        -------
        float
            Next parameter value to try
        """
        pass

    def update(self, new_value: float, new_loss: float):
        """Update agent state after evaluating a new parameter value."""
        self.current_value = new_value
        self.current_loss = new_loss
        self.history_values.append(new_value)
        self.history_losses.append(new_loss)

        if new_loss < self.best_loss:
            self.best_loss = new_loss
            self.best_value = new_value

        self.iteration += 1

    def observation(self) -> AgentObservation:
        """Get current observation for visualization."""
        return AgentObservation(
            iteration=self.iteration,
            current_value=self.current_value or 0.0,
            current_loss=self.current_loss,
            best_value=self.best_value or 0.0,
            best_loss=self.best_loss,
            history_values=list(self.history_values),
            history_losses=list(self.history_losses),
        )

    def reset(self):
        """Reset agent state."""
        self.iteration = 0
        self.history_values = []
        self.history_losses = []
        self.best_value = None
        self.best_loss = float("inf")
        self.current_value = None
        self.current_loss = float("inf")
