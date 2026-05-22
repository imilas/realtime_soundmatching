"""
Multi-dimensional agent base class.

Agents operate on a normalized [0, 1]^d space internally. The runner is
responsible for denormalizing into real parameter values when rendering audio.
This keeps step sizes meaningful across heterogeneous parameter ranges (e.g.
amp in [0.1, 1.0] vs. carrier in [10, 1000]).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Bounds:
    """Per-dimension lower/upper bounds in real parameter space."""
    lowers: np.ndarray
    uppers: np.ndarray
    names: list[str]

    def __post_init__(self):
        self.lowers = np.asarray(self.lowers, dtype=np.float64)
        self.uppers = np.asarray(self.uppers, dtype=np.float64)
        if self.lowers.shape != self.uppers.shape:
            raise ValueError("lowers/uppers shape mismatch")

    @property
    def d(self) -> int:
        return len(self.lowers)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        rng = self.uppers - self.lowers
        rng = np.where(rng == 0, 1.0, rng)
        return (x - self.lowers) / rng

    def denormalize(self, u: np.ndarray) -> np.ndarray:
        return self.lowers + u * (self.uppers - self.lowers)

    def clip_norm(self, u: np.ndarray) -> np.ndarray:
        return np.clip(u, 0.0, 1.0)


class MultiDimAgentBase(ABC):
    """
    Vector-valued optimization agent.

    Lifecycle each iteration:
      1. runner calls `propose()` to get a candidate point in normalized [0,1]^d
      2. runner denormalizes, renders audio, computes loss
      3. runner calls `observe(candidate_norm, loss)` to feed the result back

    Agents track their own internal best/history.
    """

    def __init__(self, bounds: Bounds, seed: Optional[int] = None):
        self.bounds = bounds
        self.rng = np.random.default_rng(seed)
        self.iteration = 0
        self.history_x: list[np.ndarray] = []  # normalized
        self.history_loss: list[float] = []
        self.best_x: Optional[np.ndarray] = None
        self.best_loss: float = float("inf")

    @abstractmethod
    def propose(self) -> np.ndarray:
        """Return next candidate in normalized [0,1]^d space."""
        ...

    def observe(self, x_norm: np.ndarray, loss: float) -> None:
        """Record evaluation result. Subclasses can override to learn from it."""
        x_norm = np.asarray(x_norm, dtype=np.float64)
        self.history_x.append(x_norm)
        self.history_loss.append(float(loss))
        if loss < self.best_loss:
            self.best_loss = float(loss)
            self.best_x = x_norm.copy()
        self.iteration += 1

    def best_real(self) -> Optional[np.ndarray]:
        """Best point so far in real parameter space, or None if no observations."""
        if self.best_x is None:
            return None
        return self.bounds.denormalize(self.best_x)
