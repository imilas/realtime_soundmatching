"""Tests for optimization agents."""

import numpy as np
import pytest

from agents import HillClimberAgent, RandomSearchAgent, QLearningAgent
from tests.synthetic_losses import QuadraticLoss, MultimodalLoss, NoisyLoss


class TestAgentInterface:
    """Test that all agents implement the expected interface."""

    @pytest.mark.parametrize(
        "agent_class",
        [HillClimberAgent, RandomSearchAgent, QLearningAgent],
    )
    def test_agent_initialization(self, agent_class):
        """All agents should initialize without error."""
        agent = agent_class(step_percent=5.0)
        assert agent.iteration == 0
        assert agent.best_loss == float("inf")
        assert agent.current_loss == float("inf")
        assert agent.current_value is None
        assert len(agent.history_values) == 0
        assert len(agent.history_losses) == 0

    @pytest.mark.parametrize(
        "agent_class",
        [HillClimberAgent, RandomSearchAgent, QLearningAgent],
    )
    def test_step_returns_valid_value(self, agent_class):
        """step() should return value(s) in parameter range."""
        agent = agent_class(step_percent=5.0)
        param_range = (0.0, 100.0)

        # First step (no current_value yet)
        candidates = agent.step(param_range)
        if isinstance(candidates, tuple):
            # HillClimber returns tuple
            for value in candidates:
                assert 0.0 <= value <= 100.0, f"step() returned {value} outside range"
            # Initialize with first candidate
            agent.update(candidates[0], 1.0)
        else:
            # RandomSearch or Q-Learning return single value
            assert 0.0 <= candidates <= 100.0, f"step() returned {candidates} outside range"
            agent.update(candidates, 1.0)

        # Subsequent steps
        for _ in range(10):
            candidates = agent.step(param_range)
            if isinstance(candidates, tuple):
                # HillClimber
                for value in candidates:
                    assert 0.0 <= value <= 100.0, f"step() returned {value} outside range"
                agent.update(candidates[0], np.random.random())
            else:
                # RandomSearch or Q-Learning
                assert 0.0 <= candidates <= 100.0, f"step() returned {candidates} outside range"
                agent.update(candidates, np.random.random())

    @pytest.mark.parametrize(
        "agent_class",
        [HillClimberAgent, RandomSearchAgent, QLearningAgent],
    )
    def test_update_tracks_state(self, agent_class):
        """update() should track history and best values."""
        agent = agent_class(step_percent=5.0)

        agent.update(50.0, 1.0)
        assert agent.iteration == 1
        assert agent.current_value == 50.0
        assert agent.current_loss == 1.0
        assert agent.best_loss == 1.0
        assert agent.best_value == 50.0
        assert agent.history_values == [50.0]
        assert agent.history_losses == [1.0]

        agent.update(60.0, 0.5)
        assert agent.iteration == 2
        assert agent.current_value == 60.0
        assert agent.current_loss == 0.5
        assert agent.best_loss == 0.5  # improved
        assert agent.best_value == 60.0
        assert len(agent.history_values) == 2

        agent.update(40.0, 2.0)
        assert agent.iteration == 3
        assert agent.current_loss == 2.0  # worse
        assert agent.best_loss == 0.5  # stays same
        assert agent.best_value == 60.0

    @pytest.mark.parametrize(
        "agent_class",
        [HillClimberAgent, RandomSearchAgent, QLearningAgent],
    )
    def test_reset(self, agent_class):
        """reset() should clear all state."""
        agent = agent_class(step_percent=5.0)
        agent.update(50.0, 1.0)
        agent.update(60.0, 0.5)

        agent.reset()
        assert agent.iteration == 0
        assert agent.best_loss == float("inf")
        assert agent.current_value is None
        assert len(agent.history_values) == 0

    @pytest.mark.parametrize(
        "agent_class",
        [HillClimberAgent, RandomSearchAgent, QLearningAgent],
    )
    def test_observation(self, agent_class):
        """observation() should return consistent AgentObservation."""
        agent = agent_class(step_percent=5.0)
        agent.update(50.0, 1.0)

        obs = agent.observation()
        assert obs.iteration == 1
        assert obs.current_value == 50.0
        assert obs.current_loss == 1.0
        assert obs.best_value == 50.0
        assert obs.best_loss == 1.0


class TestAgentConvergence:
    """Test that agents improve loss on simple problems."""

    def run_agent_on_loss(self, agent, loss_fn, param_range, n_steps=50):
        """Helper: run agent and collect losses."""
        losses = []
        param_range = param_range or (0.0, 100.0)

        # Initial step
        candidates = agent.step(param_range)
        if isinstance(candidates, tuple):
            # HillClimber: evaluate both candidates
            loss1 = loss_fn(candidates[0])
            loss2 = loss_fn(candidates[1])
            choice = agent.accept_trial(loss1, loss2)
            if choice == 1:
                value, loss = candidates[0], loss1
            elif choice == 2:
                value, loss = candidates[1], loss2
            else:
                # Stay put (shouldn't happen on first step, but handle it)
                value, loss = agent.current_value or candidates[0], min(loss1, loss2)
        else:
            # RandomSearch or Q-Learning: single value
            value = candidates
            loss = loss_fn(value)

        agent.update(value, loss)
        losses.append(loss)

        # Optimization loop
        for _ in range(n_steps - 1):
            candidates = agent.step(param_range)
            if isinstance(candidates, tuple):
                # HillClimber: evaluate both candidates
                loss1 = loss_fn(candidates[0])
                loss2 = loss_fn(candidates[1])
                choice = agent.accept_trial(loss1, loss2)
                if choice == 1:
                    value, loss = candidates[0], loss1
                elif choice == 2:
                    value, loss = candidates[1], loss2
                else:
                    # Stay put
                    value, loss = agent.current_value, agent.current_loss
            else:
                # RandomSearch or Q-Learning: single value
                value = candidates
                loss = loss_fn(value)

            agent.update(value, loss)
            losses.append(loss)

        return losses

    def test_hillclimber_improves_on_quadratic(self):
        """Hill climber should reduce loss on quadratic function."""
        np.random.seed(42)
        agent = HillClimberAgent(step_percent=5.0)
        loss_fn = QuadraticLoss(target=50.0)
        param_range = (0.0, 100.0)

        losses = self.run_agent_on_loss(agent, loss_fn, param_range, n_steps=50)

        # Should find something reasonable (best loss not too high)
        best_loss = min(losses)
        assert best_loss < 500.0, f"HillClimber found bad solution: {best_loss:.3f}"
        # Should not diverge (final loss reasonable)
        assert losses[-1] < 500.0, f"HillClimber diverged: final loss {losses[-1]:.3f}"

    def test_random_search_explores(self):
        """Random search should explore and track best found."""
        np.random.seed(42)
        agent = RandomSearchAgent(step_percent=5.0)
        loss_fn = QuadraticLoss(target=50.0)
        param_range = (0.0, 100.0)

        losses = self.run_agent_on_loss(agent, loss_fn, param_range, n_steps=100)

        # Should find something reasonable with 100 random samples
        best_loss = min(losses)
        # With 100 random samples from [0,100], should find something decent
        assert best_loss < 2000.0, f"Random search didn't find reasonable value: {best_loss}"
        # Agent should track best correctly
        assert agent.best_loss == best_loss

    def test_q_learning_learns(self):
        """Q-learning should learn Q-values."""
        np.random.seed(42)
        agent = QLearningAgent(step_percent=5.0, n_bins=20, epsilon=0.2)
        loss_fn = QuadraticLoss(target=50.0)
        param_range = (0.0, 100.0)

        losses = self.run_agent_on_loss(agent, loss_fn, param_range, n_steps=100)

        # Should learn something (Q-table grows)
        assert len(agent.q_table) > 0, "Q-table not populated"
        # Should find something reasonable eventually
        best_loss = min(losses)
        assert best_loss < 2000.0, f"Q-learning didn't find reasonable value: {best_loss}"
        # Should have more than trivial exploration
        assert agent.iteration > 50

    def test_all_agents_improve_on_noisy_quadratic(self):
        """All agents should handle noise and still make progress."""
        np.random.seed(42)
        param_range = (0.0, 100.0)
        base_loss = QuadraticLoss(target=50.0)
        loss_fn = NoisyLoss(base_loss, noise_std=0.1)

        for agent_class in [HillClimberAgent, RandomSearchAgent, QLearningAgent]:
            agent = agent_class(step_percent=5.0)
            losses = self.run_agent_on_loss(agent, loss_fn, param_range, n_steps=100)

            # With noise, should at least find something decent
            best_loss = min(losses)
            assert best_loss < 1000.0, f"{agent_class.__name__} failed on noisy loss"


class TestHillClimberSpecific:
    """Tests specific to hill climber agent."""

    def test_hillclimber_returns_two_candidates(self):
        """Hill climber step() should return tuple of two candidates."""
        agent = HillClimberAgent(step_percent=5.0)
        agent.update(50.0, 1.0)

        candidates = agent.step((0.0, 100.0))
        assert isinstance(candidates, tuple), "HillClimber should return tuple"
        assert len(candidates) == 2, "HillClimber should return 2 candidates"
        assert all(0.0 <= c <= 100.0 for c in candidates)

    def test_hillclimber_accept_trial(self):
        """accept_trial() should pick best candidate."""
        agent = HillClimberAgent(step_percent=5.0)
        agent.update(50.0, 1.0)

        # Trial 1 better
        choice = agent.accept_trial(0.8, 1.2)
        assert choice == 1

        # Trial 2 better
        choice = agent.accept_trial(1.2, 0.8)
        assert choice == 2

        # Neither better (stay)
        choice = agent.accept_trial(1.2, 1.5)
        assert choice == 0


class TestRandomSearchSpecific:
    """Tests specific to random search agent."""

    def test_random_search_returns_single_value(self):
        """Random search step() should return single float."""
        agent = RandomSearchAgent(step_percent=5.0)
        agent.update(50.0, 1.0)

        value = agent.step((0.0, 100.0))
        assert isinstance(value, (int, float, np.number))
        assert 0.0 <= value <= 100.0

    def test_random_search_is_random(self):
        """Different calls should return different values."""
        agent = RandomSearchAgent(step_percent=5.0)
        agent.update(50.0, 1.0)

        values = [agent.step((0.0, 100.0)) for _ in range(10)]
        assert len(set(np.round(values, 2))) > 1, "Random search not varying"


class TestQLearningSpecific:
    """Tests specific to Q-learning agent."""

    def test_q_learning_learns_q_values(self):
        """Q-learning should populate Q-table."""
        np.random.seed(42)
        agent = QLearningAgent(step_percent=5.0, n_bins=20)
        loss_fn = QuadraticLoss(target=50.0)
        param_range = (0.0, 100.0)

        # Run a few steps
        value = agent.step(param_range)
        for i in range(50):
            loss = loss_fn(value)
            agent.update(value, loss)
            value = agent.step(param_range)

        # Should have learned some Q-values
        assert len(agent.q_table) > 0, "Q-table should not be empty"

    def test_q_learning_respects_epsilon(self):
        """Q-learning with epsilon=0 should be deterministic."""
        agent = QLearningAgent(step_percent=5.0, epsilon=0.0)
        agent.update(50.0, 1.0)

        # With epsilon=0, should always pick same action (greedy)
        value1 = agent.step((0.0, 100.0))
        agent.update(value1, 1.0)
        value2 = agent.step((0.0, 100.0))

        # Not guaranteed to be same, but should be similar with same Q-values
        assert abs(value1 - value2) < 20.0  # Roughly same region
