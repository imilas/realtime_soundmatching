"""Synthetic loss functions for testing agents without JACK."""

import numpy as np


class QuadraticLoss:
    """Simple quadratic loss: (param - target)^2"""

    def __init__(self, target: float = 50.0, param_min: float = 0.0, param_max: float = 100.0):
        self.target = target
        self.param_min = param_min
        self.param_max = param_max

    def __call__(self, param: float) -> float:
        return (param - self.target) ** 2

    def __repr__(self) -> str:
        return f"QuadraticLoss(target={self.target}, range=[{self.param_min}, {self.param_max}])"


class MultimodalLoss:
    """Loss with multiple local minima to test exploration."""

    def __init__(self, param_min: float = 0.0, param_max: float = 100.0):
        self.param_min = param_min
        self.param_max = param_max

    def __call__(self, param: float) -> float:
        # Normalize to [-π, π]
        normalized = (param - self.param_min) / (self.param_max - self.param_min) * 2 * np.pi
        # Create multiple peaks
        return 1.0 - 0.5 * np.cos(normalized) - 0.3 * np.cos(2 * normalized)

    def __repr__(self) -> str:
        return f"MultimodalLoss(range=[{self.param_min}, {self.param_max}])"


class NoisyLoss:
    """Wraps another loss and adds Gaussian noise."""

    def __init__(self, base_loss, noise_std: float = 0.05, seed: int = 42):
        self.base_loss = base_loss
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

    def __call__(self, param: float) -> float:
        base = self.base_loss(param)
        noise = self.rng.normal(0, self.noise_std)
        return max(0.0, base + noise)  # Ensure non-negative

    def __repr__(self) -> str:
        return f"NoisyLoss({self.base_loss}, noise_std={self.noise_std})"
