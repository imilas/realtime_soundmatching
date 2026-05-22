"""Multi-dim uniform random search baseline."""

from __future__ import annotations

import numpy as np

from .base import Bounds, MultiDimAgentBase


class MultiDimRandomSearch(MultiDimAgentBase):
    """Each step is an independent uniform sample over [0,1]^d."""

    def __init__(self, bounds: Bounds, seed: int | None = None):
        super().__init__(bounds, seed=seed)

    def propose(self) -> np.ndarray:
        return self.rng.uniform(0.0, 1.0, size=self.bounds.d)
