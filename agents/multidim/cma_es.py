"""
CMA-ES wrapper.

The `cma` library is generation-based (ask → evaluate population → tell), which
doesn't fit MultiDimAgentBase's one-proposal-one-observation interface
directly. We buffer one generation worth of asks/tells and refill on demand.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import cma
import numpy as np

from .base import Bounds, MultiDimAgentBase


class CMAESAgent(MultiDimAgentBase):
    """
    CMA-ES operating in normalized [0,1]^d space.

    Parameters
    ----------
    bounds : Bounds
        Used for normalization; internally CMA-ES sees a [0,1]^d box.
    sigma0 : float
        Initial standard deviation in normalized space. 0.3 covers most of the
        unit box without being so wide that initial samples concentrate at the
        corners after clipping.
    population_size : int | None
        Override CMA-ES's default population (`4 + 3*log(d)`).
    """

    def __init__(
        self,
        bounds: Bounds,
        sigma0: float = 0.3,
        population_size: int | None = None,
        seed: int | None = None,
    ):
        super().__init__(bounds, seed=seed)
        opts = {
            "bounds": [[0.0] * bounds.d, [1.0] * bounds.d],
            "verbose": -9,            # silence stdout
            "verb_disp": 0,
            "verb_log": 0,
            "tolfun": 1e-12,          # let the eval budget be the only stopping criterion
            "tolx": 1e-12,
        }
        if seed is not None:
            opts["seed"] = int(seed) + 1  # cma rejects seed=0
        if population_size is not None:
            opts["popsize"] = int(population_size)

        x0 = np.full(bounds.d, 0.5)
        self._es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        self._pending_proposals: deque[np.ndarray] = deque()
        self._gen_losses: list[float] = []
        self._gen_xs: list[np.ndarray] = []

    def _refill(self) -> None:
        new_xs = self._es.ask()
        self._pending_proposals.extend(np.asarray(x, dtype=np.float64) for x in new_xs)

    def propose(self) -> np.ndarray:
        if not self._pending_proposals:
            self._refill()
        return self._pending_proposals.popleft()

    def observe(self, x_norm: np.ndarray, loss: float) -> None:
        super().observe(x_norm, loss)
        self._gen_xs.append(np.asarray(x_norm, dtype=np.float64))
        self._gen_losses.append(float(loss))
        # When we've gathered one generation worth, tell CMA-ES and reset.
        if len(self._gen_losses) == self._es.popsize:
            self._es.tell(self._gen_xs, self._gen_losses)
            self._gen_xs.clear()
            self._gen_losses.clear()
