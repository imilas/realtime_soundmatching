"""
Paper experiment configuration.

Single source of truth for synths, methods, budgets, and hyperparameters.
"""

from __future__ import annotations

from agents.multidim import (
    BayesianOptAgent,
    Bounds,
    CMAESAgent,
    MultiDimHillClimber,
    MultiDimQLearning,
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
# add_sinesaw: SIMSE_Spec (JTFS excluded from this project; spectral shape
# captures tonal sine+saw content adequately).
SYNTH_LOSS = {
    "bandpass_noise": "SIMSE_Spec",
    "sine_saw":       "SIMSE_Spec",
    "am_noise":       "DTW_Envelope",
    "sine_mod_saw":   "SIMSE_Spec",
    "sine_mod_sine":  "SIMSE_Spec",
    "add_sinesaw":    "SIMSE_Spec",
}

N_TRIALS    = 300
EVAL_BUDGET = 400
AUDIO_DURATION_S = 1.0
SAMPLE_RATE = 44100

GD_LEARNING_RATE = 0.045

# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------

def _make_rs(bounds: Bounds, seed: int) -> MultiDimRandomSearch:
    return MultiDimRandomSearch(bounds, seed=seed)


def _make_hc(bounds: Bounds, seed: int) -> MultiDimHillClimber:
    return MultiDimHillClimber(bounds, step_size=0.05, seed=seed)


def _make_cma(bounds: Bounds, seed: int) -> CMAESAgent:
    return CMAESAgent(bounds, sigma0=0.3, seed=seed)


def _make_bo(bounds: Bounds, seed: int) -> BayesianOptAgent:
    return BayesianOptAgent(bounds, n_initial_points=10, seed=seed)


# QL factory is intentionally absent — the runner manages the persistent agent.
# Maps method name → (is_gd, is_ql, agent_factory_or_None)
METHODS: dict[str, tuple[bool, bool, object]] = {
    "GD":           (True,  False, None),
    "HillClimber":  (False, False, _make_hc),
    "RandomSearch": (False, False, _make_rs),
    "CMA-ES":       (False, False, _make_cma),
    "BO":           (False, False, _make_bo),
    "QL":           (False, True,  None),
}

QL_N_BINS = 10
