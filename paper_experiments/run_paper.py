"""
Paper experiment runner.

One process per (synth, method) cell — designed for GNU parallel:

    parallel python paper_experiments/run_paper.py --synth {1} --loss {2} --method {3} --trials 200 \
        ::: bandpass_noise am_noise add_sinesaw \
        ::: "SIMSE_Spec" "DTW_Envelope" \
        ::: GD RandomSearch CMA-ES BO

Each default-loss run writes/updates results/{synth}_{method}.pkl.
Each explicit-loss run writes/updates results/{synth}_{loss}_{method}.pkl.
Resume: existing trials are loaded and counted; only missing trials are run.

Single synth/method invocation:
    python paper_experiments/run_paper.py --synth bandpass_noise --method CMA-ES --trials 200
    python paper_experiments/run_paper.py --synth bandpass_noise --loss "L1 Signal" --method CMA-ES --trials 200
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "mpl"))

from paper_experiments.config import (
    AUDIO_DURATION_S,
    GD_LEARNING_RATE,
    METHODS,
    N_TRIALS,
    SAMPLE_RATE,
    SYNTH_LOSS,
    SYNTHS,
)
from utils.loss_functions import ALL_LOSSES

RESULTS_DIR = Path(__file__).parent / "results"
GD_SUPPORTED_LOSSES = {"SIMSE_Spec", "DTW_Envelope", "JTFS"}


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


# ---------------------------------------------------------------------------
# PKL helpers
# ---------------------------------------------------------------------------

def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def _pkl_path(synth: str, method: str, loss_name: str | None = None) -> Path:
    if loss_name is None:
        return RESULTS_DIR / f"{synth}_{method}.pkl"
    return RESULTS_DIR / f"{synth}_{_slug(loss_name)}_{method}.pkl"


def _load_pkl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "rb") as f:
        return pickle.load(f)["trials"]


def _save_pkl(path: Path, trials: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"trials": trials}, f)


# ---------------------------------------------------------------------------
# Single-trial runners
# ---------------------------------------------------------------------------

def _run_gradfree(
    synth: str,
    method: str,
    factory,
    trial_idx: int,
    budget: int,
    loss_name: str,
) -> SavedTrial:
    from experiments.multidim_runner import run_trial
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
        program=synth, method=method, loss_name=loss_name,
        true_params=r.true_params, init_params=r.init_params,
        best_params=r.best_params, best_p_loss=r.best_p_loss, eval_budget=r.eval_budget,
        history_params=r.history_params, history_audio_loss=r.history_audio_loss,
        history_p_loss=r.history_p_loss, duration_s=time.perf_counter() - _t0,
    )


def _run_gd(synth: str, trial_idx: int, budget: int, loss_name: str) -> SavedTrial:
    from experiments.multidim_runner import run_trial_gd
    if loss_name not in GD_SUPPORTED_LOSSES:
        supported = ", ".join(sorted(GD_SUPPORTED_LOSSES))
        raise ValueError(f"GD does not support loss {loss_name!r}; supported: {supported}")
    _t0 = time.perf_counter()
    r = run_trial_gd(
        program_name=synth, method_name="GD", seed=trial_idx, eval_budget=budget,
        learning_rate=GD_LEARNING_RATE, sample_rate=SAMPLE_RATE,
        audio_duration_s=AUDIO_DURATION_S, loss_name=loss_name,
    )
    return SavedTrial(
        program=synth, method="GD", loss_name=loss_name,
        true_params=r.true_params, init_params=r.init_params,
        best_params=r.best_params, best_p_loss=r.best_p_loss, eval_budget=r.eval_budget,
        history_params=r.history_params, history_audio_loss=r.history_audio_loss,
        history_p_loss=r.history_p_loss, duration_s=time.perf_counter() - _t0,
    )


def _log_trial(trial_num, total, synth, method, best_p_loss, duration_s, pkl_path) -> None:
    print(f"[{trial_num}/{total}] {method} | {synth} | p_loss={best_p_loss:.4f} | "
          f"{duration_s:.1f}s | {pkl_path.name}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth",  required=True, choices=SYNTHS)
    parser.add_argument("--method", required=True, choices=list(METHODS))
    parser.add_argument(
        "--loss",
        choices=sorted(set(ALL_LOSSES) | GD_SUPPORTED_LOSSES),
        default=None,
        help="Loss to optimize. Defaults to the paper's per-synth loss.",
    )
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--budget", type=int, default=200)
    args = parser.parse_args()

    synth, method = args.synth, args.method
    loss_name = args.loss or SYNTH_LOSS[synth]
    pkl_path = _pkl_path(synth, method, args.loss)

    existing_trials = _load_pkl(pkl_path)
    n_done = len(existing_trials)
    if n_done >= args.trials:
        print(f"Already have {n_done} trials for {synth}/{method}, nothing to do.")
        return

    n_remaining = args.trials - n_done
    print(
        f"{synth}/{loss_name}/{method}: {n_done} done, "
        f"running {n_remaining} more → {pkl_path}"
    )

    is_gd, factory = METHODS[method]
    new_trials = []
    for i in range(n_remaining):
        trial_idx = n_done + i
        t = _run_gd(synth, trial_idx, args.budget, loss_name) if is_gd else \
            _run_gradfree(synth, method, factory, trial_idx, args.budget, loss_name)
        new_trials.append(t)
        _log_trial(trial_idx + 1, args.trials, synth, method, t.best_p_loss, t.duration_s, pkl_path)
        _save_pkl(pkl_path, existing_trials + [asdict(_t) for _t in new_trials])

    print(f"Saved {n_done + len(new_trials)} trials to {pkl_path}")


if __name__ == "__main__":
    main()
