"""
Paper experiment configuration.

Single source of truth for synths, methods, budgets, and hyperparameters.
Paper scope: GD vs gradient-free black-box methods (CMA-ES, RandomSearch).
"""

from __future__ import annotations

from agents.multidim import (
    Bounds,
    CMAESAgent,
    LESAgent,
    MultiDimRandomSearch,
)

# ---------------------------------------------------------------------------
# Experiment dimensions
# ---------------------------------------------------------------------------

SYNTHS = [
    "bandpass_noise",
    "am_noise",
    "add_sinesaw",
    "sine_mod_saw",
    "chirplet",
    # DX7-style FM synths (fully differentiable), one per operator routing
    "dx7_alg1",
    "dx7_alg2",
    "dx7_alg3",
]

SYNTH_LOSS = {
    "bandpass_noise":    "SIMSE_Spec",
    "am_noise":          "DTW_Envelope",
    "add_sinesaw":       "SIMSE_Spec",
    "sine_mod_saw":      "JTFS",
    "chirplet":          "JTFS",
    "dx7_alg1":          "SIMSE_Spec",
    "dx7_alg2":          "SIMSE_Spec",
    "dx7_alg3":          "SIMSE_Spec",
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


def _make_les(bounds: Bounds, seed: int) -> LESAgent:
    return LESAgent(bounds, seed=seed)


# Maps method name → (is_gd, agent_factory_or_None)
METHODS: dict[str, tuple[bool, object]] = {
    "GD":             (True,  None),
    "RandomSearch":   (False, _make_rs),
    "CMA-ES":         (False, _make_cma),
    "LES":            (False, _make_les),
}
