"""
Multi-dimensional offline experiment runner for the paper benchmark.

Compared to the existing 1D `runner.py`:
  - operates on full parameter vectors (no frozen-params requirement)
  - offline render only (no JACK/OSC); GD is added later via a separate path
  - budget is in synth evaluations, not iterations
  - decouples target generation from the optimization loop

One `run_trial(...)` call = one independent trial (one random target, one
agent run). For the paper, we run 300 trials per (synth, method) pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from agent.params import FaustParams
from agents.multidim.base import Bounds, MultiDimAgentBase
from synths.build import prepare
from utils.faust_renderer import render
from utils.loss_functions import ALL_LOSSES

# Per-synth GD loss assignments (mirrored in agents/multidim/gradient_descent.py).
SYNTH_LOSS_MAP: dict[str, str] = {
    "bandpass_noise": "SIMSE_Spec",
    "sine_saw": "JTFS",
    "am_noise": "DTW_Envelope",
    "sine_mod_saw": "JTFS",
    "sine_mod_sine": "JTFS",
    "add_sinesaw": "SIMSE_Spec",
    "chirplet": "JTFS",
    "chirplet_pulse": "DTW_Envelope",
}


@dataclass
class TrialResult:
    program_name: str
    method_name: str
    seed: int
    loss_name: str
    true_params: dict[str, float]
    init_params: dict[str, float]
    final_params: dict[str, float]
    best_params: dict[str, float]
    best_audio_loss: float
    best_p_loss: float
    history_params: list[dict[str, float]]  # per-evaluation real parameter values
    history_audio_loss: list[float]  # per-evaluation
    history_p_loss: list[float]      # per-evaluation
    eval_budget: int


def _bounds_from_params(params: FaustParams) -> Bounds:
    lowers, uppers = params.bounds()
    return Bounds(
        lowers=np.asarray(lowers, dtype=np.float64),
        uppers=np.asarray(uppers, dtype=np.float64),
        names=params.names(),
    )


def _p_loss(true_norm: np.ndarray, x_norm: np.ndarray) -> float:
    """Euclidean distance in normalized parameter space (matches IEEE paper)."""
    return float(np.linalg.norm(true_norm - x_norm))


def _render_audio(
    dsp_path: str,
    params_dict: dict[str, float],
    n_samples: int,
    sample_rate: int,
) -> np.ndarray:
    duration_s = n_samples / sample_rate
    audio = render(dsp_path, params=params_dict, duration_s=duration_s, sample_rate=sample_rate)
    if len(audio) < n_samples:
        out = np.zeros(n_samples, dtype=np.float64)
        out[: len(audio)] = audio
        return out
    return audio[:n_samples].astype(np.float64)


def run_trial(
    program_name: str,
    agent_factory: Callable[[Bounds, int], MultiDimAgentBase],
    method_name: str,
    seed: int,
    loss_name: str = "Multi-Res Spectral",
    eval_budget: int = 400,
    sample_rate: int = 44100,
    audio_duration_s: float = 1.0,
    repo_root: Optional[Path] = None,
) -> TrialResult:
    """Run a single trial: pick a random target, run the agent, record P-Loss per eval."""

    # Independent seeds for the runner (target+init) and the agent — otherwise
    # the agent's first random proposal would coincide with the runner's first
    # uniform draw, producing artificial exact matches at eval index 1.
    sub_seeds = np.random.SeedSequence(seed).generate_state(2)
    rng = np.random.default_rng(sub_seeds[0])
    agent_seed = int(sub_seeds[1])

    build = prepare(program_name)
    params = FaustParams(str(build.json_path))
    bounds = _bounds_from_params(params)

    # Random target params, uniform in real space.
    true_norm = rng.uniform(0.0, 1.0, size=bounds.d)
    true_real = bounds.denormalize(true_norm)
    true_params = params.vector_to_dict(true_real)

    # Random initial params, also uniform (independent of target).
    init_norm = rng.uniform(0.0, 1.0, size=bounds.d)
    init_real = bounds.denormalize(init_norm)
    init_params = params.vector_to_dict(init_real)

    n_samples = int(audio_duration_s * sample_rate)
    target_audio = _render_audio(str(build.dsp_path), true_params, n_samples, sample_rate)

    loss_fn = ALL_LOSSES[loss_name]
    agent = agent_factory(bounds, agent_seed)

    # Seed the agent with the initial point so its `propose()` has context if
    # it relies on a "current" state (e.g. hill climber). We pre-evaluate the
    # initial point as eval 0; if an agent ignores observations on its first
    # propose, this still counts toward the budget.
    history_audio_loss: list[float] = []
    history_p_loss: list[float] = []
    history_params: list[dict[str, float]] = []

    def evaluate(x_norm: np.ndarray) -> float:
        x_norm_c = bounds.clip_norm(x_norm)
        x_real = bounds.denormalize(x_norm_c)
        params_dict = params.vector_to_dict(x_real)
        audio = _render_audio(str(build.dsp_path), params_dict, n_samples, sample_rate)
        m = min(len(audio), len(target_audio))
        audio_loss = float(loss_fn(target_audio[:m], audio[:m], sample_rate=sample_rate))
        agent.observe(x_norm_c, audio_loss)
        history_params.append(params_dict)
        history_audio_loss.append(audio_loss)
        history_p_loss.append(_p_loss(true_norm, x_norm_c))
        return audio_loss

    # Eval 0: the initial point. Forces agent to observe it before proposing.
    evaluate(init_norm)

    while agent.iteration < eval_budget:
        candidate = agent.propose()
        evaluate(candidate)

    best_real = bounds.denormalize(agent.best_x)
    last_real = bounds.denormalize(agent.history_x[-1])

    return TrialResult(
        program_name=program_name,
        method_name=method_name,
        seed=seed,
        loss_name=loss_name,
        true_params=true_params,
        init_params=init_params,
        final_params=params.vector_to_dict(last_real),
        best_params=params.vector_to_dict(best_real),
        best_audio_loss=agent.best_loss,
        best_p_loss=_p_loss(true_norm, agent.best_x),
        history_params=history_params,
        history_audio_loss=history_audio_loss,
        history_p_loss=history_p_loss,
        eval_budget=eval_budget,
    )


def run_trial_gd(
    program_name: str,
    method_name: str = "GD",
    seed: int = 0,
    eval_budget: int = 400,
    learning_rate: float = 0.045,
    sample_rate: int = 44100,
    audio_duration_s: float = 1.0,
    loss_name: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> TrialResult:
    """
    Run a single GD trial.

    Uses the same target/init randomization as run_trial() for fair comparison.
    loss_name defaults to per-synth best from SYNTH_LOSS_MAP if not provided.
    Requires JAX, Flax, Optax, DawDreamer, kymatio, dm_pix.
    """
    from agents.multidim.gradient_descent import run_gd

    if loss_name is None:
        loss_name = SYNTH_LOSS_MAP.get(program_name, "SIMSE_Spec")

    sub_seeds = np.random.SeedSequence(seed).generate_state(2)
    rng = np.random.default_rng(sub_seeds[0])
    agent_seed = int(sub_seeds[1])

    from synths.program import get_program

    build = prepare(program_name)
    params = FaustParams(str(build.json_path))
    bounds = _bounds_from_params(params)
    program = get_program(program_name)

    true_norm = rng.uniform(0.0, 1.0, size=bounds.d)
    true_real = bounds.denormalize(true_norm)
    true_params_dict = params.vector_to_dict(true_real)

    init_norm = rng.uniform(0.0, 1.0, size=bounds.d)
    init_real = bounds.denormalize(init_norm)
    init_params_dict = params.vector_to_dict(init_real)

    n_samples = int(audio_duration_s * sample_rate)
    target_audio = _render_audio(str(build.dsp_path), true_params_dict, n_samples, sample_rate)

    # Bake init_real into the DSP code as hslider defaults so DawDreamer's Flax
    # params (which are offsets from the default, initialized to 0.0) start at
    # exactly the desired initial point.
    dsp_code = program.instantiate(init_params_dict)

    history_audio_loss, history_p_loss, history_params, best_params_real = run_gd(
        dsp_code=dsp_code,
        target_audio=target_audio,
        init_real=init_params_dict,
        param_names=params.names(),
        bounds_lowers=bounds.lowers,
        bounds_uppers=bounds.uppers,
        true_norm=true_norm,
        eval_budget=eval_budget,
        loss_name=loss_name,
        learning_rate=learning_rate,
        seed=agent_seed,
    )

    return TrialResult(
        program_name=program_name,
        method_name=method_name,
        seed=seed,
        loss_name=loss_name,
        true_params=true_params_dict,
        init_params=init_params_dict,
        final_params=best_params_real,
        best_params=best_params_real,
        best_audio_loss=min(history_audio_loss) if history_audio_loss else float("inf"),
        best_p_loss=min(history_p_loss) if history_p_loss else float("inf"),
        history_params=history_params,
        history_audio_loss=history_audio_loss,
        history_p_loss=history_p_loss,
        eval_budget=eval_budget,
    )
