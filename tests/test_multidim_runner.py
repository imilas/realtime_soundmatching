"""
Smoke test: multi-dim runner end-to-end with cheap agents.

Validates that:
  - Bounds / FaustParams integration works for an actual synth
  - The runner produces sensible TrialResults
  - HC beats RS on a small budget (sanity check the agents work)
"""

import pytest
import numpy as np

from agents.multidim import (
    BayesianOptAgent,
    Bounds,
    CMAESAgent,
    MultiDimHillClimber,
    MultiDimRandomSearch,
)
from experiments.multidim_runner import run_trial


# Tiny budget so the test runs fast even with real Faust rendering.
SMOKE_BUDGET = 20


def _factory_rs(bounds, seed):
    return MultiDimRandomSearch(bounds, seed=seed)


def _factory_hc(bounds, seed):
    return MultiDimHillClimber(bounds, step_size=0.1, seed=seed)


def _factory_cma(bounds, seed):
    # Small population so we get several "tell" cycles within SMOKE_BUDGET evals.
    return CMAESAgent(bounds, sigma0=0.3, population_size=4, seed=seed)


def _factory_bo(bounds, seed):
    # n_initial_points kept small so GP fitting kicks in within SMOKE_BUDGET.
    return BayesianOptAgent(bounds, n_initial_points=5, seed=seed)


def test_random_search_runs_end_to_end():
    result = run_trial(
        program_name="bandpass_noise",
        agent_factory=_factory_rs,
        method_name="random_search",
        seed=42,
        eval_budget=SMOKE_BUDGET,
        audio_duration_s=0.25,
    )
    assert len(result.history_audio_loss) == SMOKE_BUDGET
    assert len(result.history_p_loss) == SMOKE_BUDGET
    assert np.isfinite(result.best_audio_loss)
    assert np.isfinite(result.best_p_loss)
    # P-Loss is in normalized space (bounds = [0,1]^d), so it can't exceed sqrt(d)
    assert result.best_p_loss <= np.sqrt(2) + 1e-6
    assert set(result.best_params.keys()) == set(result.true_params.keys())


def test_cma_es_runs_end_to_end():
    result = run_trial(
        program_name="bandpass_noise",
        agent_factory=_factory_cma,
        method_name="cma_es",
        seed=42,
        eval_budget=SMOKE_BUDGET,
        audio_duration_s=0.25,
    )
    assert len(result.history_audio_loss) == SMOKE_BUDGET
    assert np.isfinite(result.best_audio_loss)
    assert result.best_p_loss <= np.sqrt(2) + 1e-6


def test_bayesian_opt_runs_end_to_end():
    result = run_trial(
        program_name="bandpass_noise",
        agent_factory=_factory_bo,
        method_name="bayesian_opt",
        seed=42,
        eval_budget=SMOKE_BUDGET,
        audio_duration_s=0.25,
    )
    assert len(result.history_audio_loss) == SMOKE_BUDGET
    assert np.isfinite(result.best_audio_loss)
    assert result.best_p_loss <= np.sqrt(2) + 1e-6


def test_hill_climber_runs_end_to_end():
    result = run_trial(
        program_name="bandpass_noise",
        agent_factory=_factory_hc,
        method_name="hill_climber",
        seed=42,
        eval_budget=SMOKE_BUDGET,
        audio_duration_s=0.25,
    )
    assert len(result.history_audio_loss) == SMOKE_BUDGET
    # Best loss is monotone non-increasing as evals proceed.
    best_so_far = np.minimum.accumulate(result.history_audio_loss)
    assert np.all(np.diff(best_so_far) <= 1e-12)
