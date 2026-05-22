"""
Bayesian optimization via scikit-optimize.

Uses `skopt.Optimizer` (the ask/tell interface) so we can feed observations
back one at a time, matching MultiDimAgentBase. Default acquisition function
is Expected Improvement (EI), a standard choice that balances exploitation
with exploration via uncertainty estimates from the GP surrogate.
"""

from __future__ import annotations

import numpy as np
from skopt import Optimizer

from .base import Bounds, MultiDimAgentBase


class BayesianOptAgent(MultiDimAgentBase):
    def __init__(
        self,
        bounds: Bounds,
        acq_func: str = "EI",
        n_initial_points: int = 10,
        seed: int | None = None,
    ):
        super().__init__(bounds, seed=seed)
        # Operate in normalized [0,1]^d (consistent with HC/RS/CMA-ES).
        dimensions = [(0.0, 1.0)] * bounds.d
        self._opt = Optimizer(
            dimensions=dimensions,
            base_estimator="GP",
            acq_func=acq_func,
            n_initial_points=n_initial_points,
            random_state=seed,
        )

    def propose(self) -> np.ndarray:
        return np.asarray(self._opt.ask(), dtype=np.float64)

    def observe(self, x_norm: np.ndarray, loss: float) -> None:
        super().observe(x_norm, loss)
        self._opt.tell(list(map(float, x_norm)), float(loss))
