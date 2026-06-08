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
    "sine_saw",
    "sine_mod_saw",
    "sine_mod_sine",
    "chirplet",
    "chirplet_pulse",
    # v1 variants: exact DSP from the old in-domain paper, used for GD verification only
    "bandpass_noise_v1",
    "am_noise_v1",
]

# Per-synth "expected best" loss from the IEEE 2025 paper (in-domain) and the
# ISMIR 2026 OOD paper. Used as the default loss when --loss isn't passed.
# chirplet and chirplet_pulse come from the OOD paper: chirplet's best loss is
# JTFS; chirplet_pulse's is DTW_Envelope (JTFS fails there — the headline finding).
SYNTH_LOSS = {
    "bandpass_noise":    "SIMSE_Spec",
    "am_noise":          "DTW_Envelope",
    "add_sinesaw":       "SIMSE_Spec",
    "sine_saw":          "JTFS",
    "sine_mod_saw":      "JTFS",
    "sine_mod_sine":     "JTFS",
    "chirplet":          "JTFS",
    "chirplet_pulse":    "DTW_Envelope",
    "bandpass_noise_v1": "SIMSE_Spec",
    "am_noise_v1":       "DTW_Envelope",
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
