"""
Q-learning agent for parameter optimization.

Discretizes parameter space into bins, learns Q-values for state-action pairs,
and uses epsilon-greedy exploration/exploitation.
"""

import numpy as np

from .base import AgentBase


class QLearningAgent(AgentBase):
    """
    Tabular Q-learning with discretized parameter space.

    State = current bin in discretized space
    Actions = move left/stay/right (discrete movements)
    """

    def __init__(
        self,
        step_percent: float = 5.0,
        n_bins: int = 20,
        epsilon: float = 0.15,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
    ):
        """
        Parameters
        ----------
        step_percent : float
            Step size as percentage of parameter range (not directly used, for compatibility)
        n_bins : int
            Number of bins to discretize parameter space into
        epsilon : float
            Exploration rate (probability of random action)
        learning_rate : float
            Q-value update rate
        discount_factor : float
            Discount factor for future rewards (0=greedy, 1=far-sighted)
        """
        super().__init__(step_percent)
        self.n_bins = n_bins
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Q-table: (state_bin, action) -> Q-value
        # Actions: 0=left, 1=stay, 2=right
        self.q_table = {}
        self.action_names = ["left", "stay", "right"]

        self.prev_state = None
        self.prev_action = None

    def _value_to_bin(self, value: float, param_range: tuple[float, float]) -> int:
        """Convert parameter value to bin index."""
        param_min, param_max = param_range
        if param_max == param_min:
            return 0
        normalized = (value - param_min) / (param_max - param_min)
        bin_idx = int(np.clip(normalized * self.n_bins, 0, self.n_bins - 1))
        return bin_idx

    def _bin_to_value(self, bin_idx: int, param_range: tuple[float, float]) -> float:
        """Convert bin index to parameter value (center of bin)."""
        param_min, param_max = param_range
        bin_width = (param_max - param_min) / self.n_bins
        return param_min + (bin_idx + 0.5) * bin_width

    def _get_q_value(self, state: int, action: int) -> float:
        """Get Q-value, defaulting to 0 for unseen state-action pairs."""
        return self.q_table.get((state, action), 0.0)

    def _set_q_value(self, state: int, action: int, value: float):
        """Set Q-value in table."""
        self.q_table[(state, action)] = value

    def step(self, param_range: tuple[float, float]) -> float:
        """Decide next parameter value using epsilon-greedy policy."""
        param_min, param_max = param_range

        if self.current_value is None:
            # First step: start in middle
            return (param_min + param_max) / 2.0

        current_bin = self._value_to_bin(self.current_value, param_range)

        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.choice(len(self.action_names))
        else:
            # Exploit: best Q-value
            q_values = [self._get_q_value(current_bin, a) for a in range(len(self.action_names))]
            action = np.argmax(q_values)

        # Convert action to next bin
        if action == 0:  # move left
            next_bin = max(0, current_bin - 1)
        elif action == 2:  # move right
            next_bin = min(self.n_bins - 1, current_bin + 1)
        else:  # stay
            next_bin = current_bin

        # Store for update in accept_trial or next step
        self.prev_state = current_bin
        self.prev_action = action

        # Convert bin to parameter value
        next_value = self._bin_to_value(next_bin, param_range)
        return float(np.clip(next_value, param_min, param_max))

    def update(self, new_value: float, new_loss: float):
        """Update agent state and learn from the step."""
        # Update Q-value if we have previous state/action
        if self.prev_state is not None and self.prev_action is not None:
            # Reward is negative loss (minimize loss = maximize reward)
            reward = -new_loss

            # Get new state
            param_range = self._infer_param_range()
            if param_range:
                new_bin = self._value_to_bin(new_value, param_range)

                # Q-learning update: Q(s,a) += lr * (r + gamma * max(Q(s',a')) - Q(s,a))
                old_q = self._get_q_value(self.prev_state, self.prev_action)
                max_next_q = max(
                    self._get_q_value(new_bin, a) for a in range(len(self.action_names))
                )
                new_q = old_q + self.learning_rate * (
                    reward + self.discount_factor * max_next_q - old_q
                )
                self._set_q_value(self.prev_state, self.prev_action, new_q)

        # Standard base class update
        super().update(new_value, new_loss)

    def _infer_param_range(self) -> tuple[float, float] | None:
        """Try to infer parameter range from history."""
        if not self.history_values:
            return None
        # Conservative estimate: use min/max of visited values with some margin
        min_val = min(self.history_values)
        max_val = max(self.history_values)
        margin = (max_val - min_val) * 0.1 if max_val > min_val else 1.0
        return (min_val - margin, max_val + margin)

    def reset(self):
        """Reset agent state and Q-table."""
        super().reset()
        self.q_table.clear()
        self.prev_state = None
        self.prev_action = None
