"""
Paper experiment configuration.

Single source of truth for synths, methods, budgets, and hyperparameters.
Paper scope: GD vs gradient-free black-box methods (CMA-ES, BO, RandomSearch).
"""

from __future__ import annotations

from agents.multidim import (
    BayesianOptAgent,
    Bounds,
    CMAESAgent,
    MultiDimRandomSearch,
)

# ---------------------------------------------------------------------------
# Experiment dimensions
# ---------------------------------------------------------------------------

SYNTHS = [
    "bandpass_noise",
    "am_noise",
    "add_sinesaw",
]

# Best-performing loss per synth (from IEEE 2025 paper), used for ALL methods.
SYNTH_LOSS = {
    "bandpass_noise": "SIMSE_Spec",
    "am_noise":       "DTW_Envelope",
    "add_sinesaw":    "SIMSE_Spec",
}

N_TRIALS    = 200
EVAL_BUDGET = 200
AUDIO_DURATION_S = 1.0
SAMPLE_RATE = 44100

GD_LEARNING_RATE = 0.045

# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------

def _make_rs(bounds: Bounds, seed: int) -> MultiDimRandomSearch:
    return MultiDimRandomSearch(bounds, seed=seed)


def _make_cma(bounds: Bounds, seed: int) -> CMAESAgent:
    return CMAESAgent(bounds, sigma0=0.3, seed=seed)


def _make_bo(bounds: Bounds, seed: int) -> BayesianOptAgent:
    return BayesianOptAgent(bounds, n_initial_points=10, seed=seed)


# Maps method name → (is_gd, agent_factory_or_None)
METHODS: dict[str, tuple[bool, object]] = {
    "GD":           (True,  None),
    "RandomSearch": (False, _make_rs),
    "CMA-ES":       (False, _make_cma),
    "BO":           (False, _make_bo),
}
