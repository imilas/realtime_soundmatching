"""
evosax-based agents (CMA-ES and Learned Evolution Strategy).

evosax is JAX-native and generation-based (ask -> evaluate population ->
tell), same shape mismatch with MultiDimAgentBase's one-proposal-one-
observation interface as the `cma` library. We buffer one generation worth
of asks/tells and refill on demand, mirroring cma_es.py.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .base import Bounds, MultiDimAgentBase


class EvosaxAgent(MultiDimAgentBase):
    """
    Generic wrapper around an evosax distribution-based algorithm, operating
    in normalized [0,1]^d space (mean initialized at the center, candidates
    clipped to the unit box before being handed out).
    """

    def __init__(
        self,
        bounds: Bounds,
        algorithm_cls,
        population_size: Optional[int] = None,
        seed: Optional[int] = None,
        **algo_kwargs,
    ):
        super().__init__(bounds, seed=seed)
        if population_size is None:
            population_size = max(4, int(4 + 3 * np.log(bounds.d)))

        solution = jnp.zeros(bounds.d)
        self._es = algorithm_cls(population_size=population_size, solution=solution, **algo_kwargs)
        self._params = self._es.default_params
        self._key = jax.random.PRNGKey(int(seed) if seed is not None else 0)
        self._key, init_key = jax.random.split(self._key)
        self._state = self._es.init(init_key, jnp.full(bounds.d, 0.5), self._params)

        self._pending_proposals: deque[np.ndarray] = deque()
        self._gen_xs: list[np.ndarray] = []
        self._gen_losses: list[float] = []
        self._popsize = population_size

    def _refill(self) -> None:
        self._key, ask_key = jax.random.split(self._key)
        xs, self._state = self._es.ask(ask_key, self._state, self._params)
        xs = np.clip(np.asarray(xs, dtype=np.float64), 0.0, 1.0)
        self._pending_proposals.extend(xs)

    def propose(self) -> np.ndarray:
        if not self._pending_proposals:
            self._refill()
        return self._pending_proposals.popleft()

    def observe(self, x_norm: np.ndarray, loss: float) -> None:
        super().observe(x_norm, loss)
        self._gen_xs.append(np.asarray(x_norm, dtype=np.float64))
        self._gen_losses.append(float(loss))
        if len(self._gen_losses) == self._popsize:
            population = jnp.asarray(np.stack(self._gen_xs), dtype=jnp.float32)
            fitness = jnp.asarray(self._gen_losses, dtype=jnp.float32)
            self._key, tell_key = jax.random.split(self._key)
            self._state, _ = self._es.tell(tell_key, population, fitness, self._state, self._params)
            self._gen_xs.clear()
            self._gen_losses.clear()


class CMAESEvosaxAgent(EvosaxAgent):
    """CMA-ES via evosax, for cross-checking against the `cma`-library wrapper."""

    def __init__(self, bounds: Bounds, population_size: Optional[int] = None, seed: Optional[int] = None):
        from evosax.algorithms import CMA_ES
        super().__init__(bounds, CMA_ES, population_size=population_size, seed=seed)


class LESAgent(EvosaxAgent):
    """Learned Evolution Strategy (Lange et al.), evosax's pretrained meta-learned ES."""

    def __init__(self, bounds: Bounds, population_size: Optional[int] = None, seed: Optional[int] = None):
        from evosax.algorithms import LearnedES
        super().__init__(bounds, LearnedES, population_size=population_size, seed=seed)
