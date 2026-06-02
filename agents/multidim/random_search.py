"""Uniform random search baseline.

Each proposal is drawn independently and uniformly from [0,1]^d, regardless of
previous observations. This is the standard random-search floor: it ignores the
objective but provides an unbiased coverage of the parameter space, unlike a
random walk whose step size limits exploration radius.
"""

from __future__ import annotations

import numpy as np

from .base import Bounds, MultiDimAgentBase


class MultiDimRandomSearch(MultiDimAgentBase):
    """Independent uniform samples — canonical random-search baseline."""

    def __init__(self, bounds: Bounds, seed: int | None = None, **_):
        super().__init__(bounds, seed=seed)

    def propose(self) -> np.ndarray:
        return self.rng.uniform(0.0, 1.0, size=self.bounds.d)

    def observe(self, x_norm: np.ndarray, loss: float) -> None:
        super().observe(x_norm, loss)
