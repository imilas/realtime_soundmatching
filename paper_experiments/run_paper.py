"""
Paper experiment runner.

One process per (synth, method) cell — designed for GNU parallel:

    parallel python paper_experiments/run_paper.py --synth {1} --method {2} --trials 10 \
        ::: bandpass_noise sine_saw am_noise sine_mod_saw sine_mod_sine \
        ::: GD HillClimber RandomSearch CMA-ES BO QL

Each run writes/updates  results/{synth}_{method}.pkl
Resume: existing trials are loaded and counted; only missing trials are run.
For QL the persisted Q-table is restored from the pkl so the policy carries over.

Single synth/method invocation:
    python paper_experiments/run_paper.py --synth bandpass_noise --method HillClimber --trials 10
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_experiments.config import (
    AUDIO_DURATION_S,
    EVAL_BUDGET,
    GD_LEARNING_RATE,
    METHODS,
    N_TRIALS,
    QL_N_BINS,
    SAMPLE_RATE,
    SYNTH_LOSS,
    SYNTHS,
)

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Saved data structure
# ---------------------------------------------------------------------------

@dataclass
class SavedTrial:
    program: str
    method: str
    loss_name: str
    true_params: dict
    init_params: dict
    best_params: dict
    best_p_loss: float
    eval_budget: int
    history_params: list[dict] = field(default_factory=list)
    history_audio_loss: list[float] = field(default_factory=list)
    history_p_loss: list[float] = field(default_factory=list)
    duration_s: float = 0.0
    ql_q_table_size: Optional[int] = None
    ql_epsilon_end: Optional[float] = None


# ---------------------------------------------------------------------------
# PKL helpers
# ---------------------------------------------------------------------------

def _pkl_path(synth: str, method: str) -> Path:
    return RESULTS_DIR / f"{synth}_{method}.pkl"


def _load_pkl(path: Path) -> tuple[list[dict], Optional[dict], Optional[float]]:
    """Returns (trials, ql_q_table, ql_epsilon). ql_* are None for non-QL."""
    if not path.exists():
        return [], None, None
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["trials"], data.get("ql_q_table"), data.get("ql_epsilon")


def _save_pkl(
    path: Path,
    trials: list[dict],
    ql_q_table: Optional[dict] = None,
    ql_epsilon: Optional[float] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"trials": trials, "ql_q_table": ql_q_table, "ql_epsilon": ql_epsilon}, f)


# ---------------------------------------------------------------------------
# Single-trial runners
# ---------------------------------------------------------------------------

def _run_gradfree(synth: str, method: str, factory, trial_idx: int, budget: int) -> SavedTrial:
    from experiments.multidim_runner import run_trial
    loss_name = SYNTH_LOSS[synth]
    _t0 = time.perf_counter()
    r = run_trial(
        program_name=synth,
        agent_factory=factory,
        method_name=method,
        seed=trial_idx,
        loss_name=loss_name,
        eval_budget=budget,
        sample_rate=SAMPLE_RATE,
        audio_duration_s=AUDIO_DURATION_S,
    )
    return SavedTrial(
        program=synth,
        method=method,
        loss_name=loss_name,
        true_params=r.true_params,
        init_params=r.init_params,
        best_params=r.best_params,
        best_p_loss=r.best_p_loss,
        eval_budget=r.eval_budget,
        history_params=r.history_params,
        history_audio_loss=r.history_audio_loss,
        history_p_loss=r.history_p_loss,
        duration_s=time.perf_counter() - _t0,
    )


def _run_gd(synth: str, trial_idx: int, budget: int) -> SavedTrial:
    from experiments.multidim_runner import run_trial_gd
    loss_name = SYNTH_LOSS[synth]
    _t0 = time.perf_counter()
    r = run_trial_gd(
        program_name=synth,
        method_name="GD",
        seed=trial_idx,
        eval_budget=budget,
        learning_rate=GD_LEARNING_RATE,
        sample_rate=SAMPLE_RATE,
        audio_duration_s=AUDIO_DURATION_S,
        loss_name=loss_name,
    )
    return SavedTrial(
        program=synth,
        method="GD",
        loss_name=loss_name,
        true_params=r.true_params,
        init_params=r.init_params,
        best_params=r.best_params,
        best_p_loss=r.best_p_loss,
        eval_budget=r.eval_budget,
        history_params=r.history_params,
        history_audio_loss=r.history_audio_loss,
        history_p_loss=r.history_p_loss,
        duration_s=time.perf_counter() - _t0,
    )


# ---------------------------------------------------------------------------
# QL multi-trial runner (persistent Q-table across trials)
# ---------------------------------------------------------------------------

def _run_ql_trials(
    synth: str,
    n_trials: int,
    start_idx: int,
    existing_q_table: Optional[dict],
    existing_epsilon: Optional[float],
    budget: int = 200,
    pkl_path: Optional[Path] = None,
    existing_trials: Optional[list[dict]] = None,
) -> tuple[list[SavedTrial], dict, float]:
    """
    Run n_trials of QL starting from trial index start_idx.
    Restores Q-table and epsilon from a previous run if provided.
    Returns (new_trials, final_q_table, final_epsilon).
    """
    from agents.multidim import MultiDimQLearning
    from agent.params import FaustParams
    from experiments.multidim_runner import _bounds_from_params, _p_loss, _render_audio
    from synths.build import prepare
    from utils.loss_functions import ALL_LOSSES

    build = prepare(synth)
    params = FaustParams(str(build.json_path))
    bounds = _bounds_from_params(params)
    loss_name = SYNTH_LOSS[synth]
    loss_fn = ALL_LOSSES[loss_name]
    n_samples = int(AUDIO_DURATION_S * SAMPLE_RATE)

    agent = MultiDimQLearning(bounds, n_bins=QL_N_BINS, seed=start_idx)
    if existing_q_table is not None:
        agent.q_table = existing_q_table
    if existing_epsilon is not None:
        agent.epsilon = existing_epsilon

    new_trials: list[SavedTrial] = []

    for i in range(n_trials):
        trial_idx = start_idx + i
        sub_seeds = np.random.SeedSequence(trial_idx).generate_state(2)
        rng = np.random.default_rng(sub_seeds[0])

        true_norm = rng.uniform(0.0, 1.0, size=bounds.d)
        true_real = bounds.denormalize(true_norm)
        true_params = params.vector_to_dict(true_real)

        init_norm = rng.uniform(0.0, 1.0, size=bounds.d)
        init_real = bounds.denormalize(init_norm)

        target_audio = _render_audio(str(build.dsp_path), true_params, n_samples, SAMPLE_RATE)
        agent.soft_reset()
        history_params: list[dict] = []
        history_audio_loss: list[float] = []
        history_p_loss: list[float] = []
        _t0 = time.perf_counter()

        def evaluate(x_norm: np.ndarray) -> float:
            x_norm_c = bounds.clip_norm(x_norm)
            x_real = bounds.denormalize(x_norm_c)
            params_dict = params.vector_to_dict(x_real)
            audio = _render_audio(str(build.dsp_path), params_dict, n_samples, SAMPLE_RATE)
            m = min(len(audio), len(target_audio))
            al = float(loss_fn(target_audio[:m], audio[:m], sample_rate=SAMPLE_RATE))
            agent.observe(x_norm_c, al)
            history_params.append(params_dict)
            history_audio_loss.append(al)
            history_p_loss.append(_p_loss(true_norm, x_norm_c))
            return al

        evaluate(init_norm)
        while agent.iteration < budget:
            evaluate(agent.propose())

        t = SavedTrial(
            program=synth,
            method="QL",
            loss_name=loss_name,
            true_params=true_params,
            init_params=params.vector_to_dict(init_real),
            best_params=params.vector_to_dict(bounds.denormalize(agent.best_x)),
            best_p_loss=_p_loss(true_norm, agent.best_x),
            eval_budget=budget,
            history_params=history_params,
            history_audio_loss=history_audio_loss,
            history_p_loss=history_p_loss,
            duration_s=time.perf_counter() - _t0,
            ql_q_table_size=agent.q_table_size,
            ql_epsilon_end=agent.epsilon,
        )
        new_trials.append(t)
        if pkl_path is not None:
            _save_pkl(
                pkl_path,
                (existing_trials or []) + [asdict(_t) for _t in new_trials],
                agent.q_table,
                agent.epsilon,
            )
        _log_trial(trial_idx + 1, n_trials + start_idx, synth, "QL", t.best_p_loss, t.duration_s, pkl_path or Path(f"{synth}_QL.pkl"))

    return new_trials, agent.q_table, agent.epsilon


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_trial(
    trial_num: int,
    total: int,
    synth: str,
    method: str,
    best_p_loss: float,
    duration_s: float,
    pkl_path: Path,
) -> None:
    print(
        f"[{trial_num}/{total}] {method} | {synth} | p_loss={best_p_loss:.4f} | {duration_s:.1f}s | {pkl_path.name}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth",   required=True, choices=SYNTHS)
    parser.add_argument("--method",  required=True, choices=list(METHODS))
    parser.add_argument("--trials",  type=int, default=N_TRIALS)
    parser.add_argument("--budget",  type=int, default=200)
    args = parser.parse_args()

    synth, method = args.synth, args.method
    pkl_path = _pkl_path(synth, method)

    existing_trials, ql_q_table, ql_epsilon = _load_pkl(pkl_path)
    n_done = len(existing_trials)

    if n_done >= args.trials:
        print(f"Already have {n_done} trials for {synth}/{method}, nothing to do.")
        return

    n_remaining = args.trials - n_done
    print(f"{synth}/{method}: {n_done} done, running {n_remaining} more → {pkl_path}")

    is_gd, is_ql, factory = METHODS[method]

    if is_ql:
        new_trials, ql_q_table, ql_epsilon = _run_ql_trials(
            synth,
            n_remaining,
            n_done,
            ql_q_table,
            ql_epsilon,
            args.budget,
            pkl_path,
            existing_trials,
        )
    else:
        new_trials = []
        for i in range(n_remaining):
            trial_idx = n_done + i
            if is_gd:
                t = _run_gd(synth, trial_idx, args.budget)
            else:
                t = _run_gradfree(synth, method, factory, trial_idx, args.budget)
            new_trials.append(t)
            _log_trial(trial_idx + 1, args.trials, synth, method, t.best_p_loss, t.duration_s, pkl_path)
            _save_pkl(
                pkl_path,
                existing_trials + [asdict(_t) for _t in new_trials],
                ql_q_table,
                ql_epsilon,
            )

    all_trials = existing_trials + [asdict(t) for t in new_trials]
    _save_pkl(pkl_path, all_trials, ql_q_table, ql_epsilon)
    print(f"Saved {len(all_trials)} trials to {pkl_path}")


if __name__ == "__main__":
    main()
