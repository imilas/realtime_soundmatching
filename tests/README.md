# Agent Tests

Fast, reproducible tests for optimization agents **without JACK or GUI**.

## Quick Start

Install test dependencies:
```bash
pip install pytest
```

Run all tests:
```bash
pytest
```

Run with verbose output:
```bash
pytest -v
```

## What's Tested

- **Interface**: All agents have correct API (step, update, reset)
- **Convergence**: Agents improve on synthetic loss functions
- **Correctness**: Algorithm-specific behavior (HillClimber has 2 candidates, etc.)

## Synthetic Loss Functions

Tests don't use JACK—instead, they use fast synthetic functions:

- **QuadraticLoss**: Simple convex problem `(param - target)^2`
- **MultimodalLoss**: Multiple local minima (tests exploration)
- **NoisyLoss**: Adds random noise (tests robustness)

This means:
- ✓ Tests run in seconds (no JACK)
- ✓ Tests are deterministic (seeded randomness)
- ✓ Tests work anywhere (no hardware)

## Test Organization

```
TestAgentInterface
  ✓ Initialization works
  ✓ step() returns valid values
  ✓ update() tracks state
  ✓ reset() clears state
  ✓ observation() works

TestAgentConvergence
  ✓ HillClimber improves on quadratic
  ✓ RandomSearch explores
  ✓ QLearning learns Q-values
  ✓ All agents handle noise

TestHillClimberSpecific
  ✓ Returns 2 candidates
  ✓ accept_trial() picks best

TestRandomSearchSpecific
  ✓ Returns single value
  ✓ Is actually random

TestQLearningSpecific
  ✓ Learns Q-values
  ✓ Respects epsilon
```

## Common Commands

```bash
# Run all tests
pytest

# Run specific test class
pytest tests/test_agents.py::TestAgentConvergence -v

# Run tests matching pattern
pytest -k "quadratic" -v

# Run and stop on first failure
pytest -x

# Run with short traceback
pytest --tb=short
```

## Adding a New Agent

1. Implement agent in `agents/my_agent.py`
2. Add tests in `test_agents.py`:
   ```python
   @pytest.mark.parametrize(
       "agent_class",
       [HillClimberAgent, RandomSearchAgent, QLearningAgent, MyNewAgent],  # ← add here
   )
   def test_agent_initialization(self, agent_class):
       ...
   ```
3. Run tests: `pytest -v`
