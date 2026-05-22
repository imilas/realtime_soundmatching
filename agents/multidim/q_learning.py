"""
Multi-dimensional tabular Q-learning agent.

State  = (position_bins..., loss_bucket)
         Position bins: current location in [0,1]^d discretized into n_bins per dim.
         Loss bucket:   current audio loss quantized into n_loss_buckets using a
                        running min/max normalization (persists across trials so the
                        scale stays consistent as the agent learns).

Actions = all 3^d combinations of {-1, 0, +1} bin steps (one per dimension).

Reward  = prev_loss - loss  (positive when we improve, negative when we worsen).
          Delta reward gives a dense signal and avoids the scale varying by synth.

Q-table and loss statistics persist across trials via soft_reset() — position
resets but the learned policy carries over.
"""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np

from .base import Bounds, MultiDimAgentBase


class MultiDimQLearning(MultiDimAgentBase):
    def __init__(
        self,
        bounds: Bounds,
        n_bins: int = 10,
        n_loss_buckets: int = 5,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        learning_rate: float = 0.1,
        discount: float = 0.9,
        seed: Optional[int] = None,
    ):
        super().__init__(bounds, seed=seed)
        self.n_bins = n_bins
        self.n_loss_buckets = n_loss_buckets
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = learning_rate
        self.discount = discount

        # All 3^d (delta_dim0, ...) combinations
        self._actions: list[tuple[int, ...]] = list(
            itertools.product([-1, 0, 1], repeat=bounds.d)
        )
        self.q_table: dict[tuple, float] = {}  # (state, action_idx) -> value

        # Running loss statistics — persist across trials for consistent bucketing.
        self._loss_min: float = float("inf")
        self._loss_max: float = float("-inf")

        self._current_state: Optional[tuple[int, ...]] = None  # (bins..., loss_bucket)
        self._prev_state: Optional[tuple[int, ...]] = None
        self._prev_action: Optional[int] = None
        self._prev_loss: Optional[float] = None
        self.trial_count: int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_bin(self, x_norm: np.ndarray) -> tuple[int, ...]:
        idx = np.clip((x_norm * self.n_bins).astype(int), 0, self.n_bins - 1)
        return tuple(int(i) for i in idx)

    def _bin_to_norm(self, pos_bins: tuple[int, ...]) -> np.ndarray:
        return (np.array(pos_bins, dtype=float) + 0.5) / self.n_bins

    def _loss_bucket(self, loss: float) -> int:
        rng = self._loss_max - self._loss_min
        if rng < 1e-8:
            return 0
        normalized = (loss - self._loss_min) / rng
        return int(np.clip(normalized * self.n_loss_buckets, 0, self.n_loss_buckets - 1))

    def _get_q(self, state: tuple, action_idx: int) -> float:
        return self.q_table.get((state, action_idx), 0.0)

    def _best_action(self, state: tuple) -> int:
        q_vals = [self._get_q(state, i) for i in range(len(self._actions))]
        return int(np.argmax(q_vals))

    # ------------------------------------------------------------------
    # MultiDimAgentBase interface
    # ------------------------------------------------------------------

    def propose(self) -> np.ndarray:
        if self._current_state is None:
            x = np.full(self.bounds.d, 0.5)
            self._prev_state = None
            self._prev_action = None
            return x

        if self.rng.random() < self.epsilon:
            action_idx = int(self.rng.integers(len(self._actions)))
        else:
            action_idx = self._best_action(self._current_state)

        self._prev_state = self._current_state
        self._prev_action = action_idx

        pos_bins = self._current_state[:-1]  # strip loss bucket
        delta = np.array(self._actions[action_idx], dtype=float) / self.n_bins
        next_norm = self.bounds.clip_norm(self._bin_to_norm(pos_bins) + delta)
        return next_norm

    def observe(self, x_norm: np.ndarray, loss: float) -> None:
        # Update running loss statistics before bucketing.
        self._loss_min = min(self._loss_min, loss)
        self._loss_max = max(self._loss_max, loss)

        new_state = self._to_bin(x_norm) + (self._loss_bucket(loss),)

        if self._prev_state is not None and self._prev_action is not None:
            # Delta reward: positive when we improve, negative when we worsen.
            reward = (self._prev_loss - loss) if self._prev_loss is not None else 0.0
            max_next_q = max(self._get_q(new_state, i) for i in range(len(self._actions)))
            old_q = self._get_q(self._prev_state, self._prev_action)
            self.q_table[(self._prev_state, self._prev_action)] = old_q + self.lr * (
                reward + self.discount * max_next_q - old_q
            )

        self._current_state = new_state
        self._prev_loss = loss
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        super().observe(x_norm, loss)

    # ------------------------------------------------------------------
    # Trial lifecycle
    # ------------------------------------------------------------------

    def soft_reset(self) -> None:
        """Reset per-trial state (position, history) but keep Q-table and loss stats."""
        self._current_state = None
        self._prev_state = None
        self._prev_action = None
        self._prev_loss = None
        self.iteration = 0
        self.history_x = []
        self.history_loss = []
        self.best_x = None
        self.best_loss = float("inf")
        self.trial_count += 1

    @property
    def q_table_size(self) -> int:
        return len(self.q_table)
