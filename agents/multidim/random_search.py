"""Multi-dim random-walk search baseline."""

from __future__ import annotations

import numpy as np

from .base import Bounds, MultiDimAgentBase


class MultiDimRandomSearch(MultiDimAgentBase):
    """Move randomly around the last observed point in normalized space."""

    def __init__(self, bounds: Bounds, step_size: float = 0.1, seed: int | None = None):
        super().__init__(bounds, seed=seed)
        self.step_size = step_size
        self._current: np.ndarray | None = None

    def propose(self) -> np.ndarray:
        if self._current is None:
            self._current = np.full(self.bounds.d, 0.5)
        noise = self.rng.normal(0.0, self.step_size, size=self.bounds.d)
        return self.bounds.clip_norm(self._current + noise)

    def observe(self, x_norm: np.ndarray, loss: float) -> None:
        super().observe(x_norm, loss)
        self._current = np.asarray(x_norm, dtype=np.float64).copy()
