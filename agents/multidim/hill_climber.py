"""
Multi-dim isotropic Gaussian hill climber.

Maintains a current point; each step samples a perturbation around it.
Move is accepted only if it improves the best loss seen so far (greedy).
"""

from __future__ import annotations

import numpy as np

from .base import Bounds, MultiDimAgentBase


class MultiDimHillClimber(MultiDimAgentBase):
    def __init__(
        self,
        bounds: Bounds,
        step_size: float = 0.05,
        seed: int | None = None,
    ):
        super().__init__(bounds, seed=seed)
        self.step_size = step_size
        self._current: np.ndarray | None = None
        self._proposed: np.ndarray | None = None

    def propose(self) -> np.ndarray:
        if self._current is None:
            # First proposal: midpoint of the box. Deterministic start gives
            # cleaner trial-to-trial comparison; randomness comes from
            # subsequent Gaussian perturbations.
            self._proposed = np.full(self.bounds.d, 0.5)
        else:
            noise = self.rng.normal(0.0, self.step_size, size=self.bounds.d)
            self._proposed = self.bounds.clip_norm(self._current + noise)
        return self._proposed

    def observe(self, x_norm: np.ndarray, loss: float) -> None:
        super().observe(x_norm, loss)
        # Greedy accept: move to the best point seen so far.
        if self._current is None or loss <= self.best_loss + 1e-12:
            self._current = np.asarray(x_norm, dtype=np.float64).copy()
