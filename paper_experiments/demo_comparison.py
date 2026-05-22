"""
Demo: HC vs RS vs CMA-ES on bandpass_noise.

Small comparison to validate the multi-dim pipeline end-to-end before scaling
to the full paper experiment (300 trials × 5 synths × 5 methods).

Run:
    python paper_experiments/demo_comparison.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running as a script: add repo root to sys.path so `agents` and
# `experiments` resolve correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from agents.multidim import (
    BayesianOptAgent,
    Bounds,
    CMAESAgent,
    MultiDimHillClimber,
    MultiDimRandomSearch,
)
from experiments.multidim_runner import run_trial


def _factory_rs(bounds: Bounds, seed: int) -> MultiDimRandomSearch:
    return MultiDimRandomSearch(bounds, seed=seed)


def _factory_hc(bounds: Bounds, seed: int) -> MultiDimHillClimber:
    return MultiDimHillClimber(bounds, step_size=0.05, seed=seed)


def _factory_cma(bounds: Bounds, seed: int) -> CMAESAgent:
    return CMAESAgent(bounds, sigma0=0.3, seed=seed)


def _factory_bo(bounds: Bounds, seed: int) -> BayesianOptAgent:
    return BayesianOptAgent(bounds, n_initial_points=10, seed=seed)


METHODS = {
    "RandomSearch": _factory_rs,
    "HillClimber":  _factory_hc,
    "CMA-ES":       _factory_cma,
    "BO":           _factory_bo,
}


def main(
    program_name: str = "bandpass_noise",
    n_trials: int = 5,
    eval_budget: int = 100,
    audio_duration_s: float = 0.5,
    loss_name: str = "Multi-Res Spectral",
) -> None:
    print(f"Synth: {program_name}   Loss: {loss_name}")
    print(f"Trials per method: {n_trials}   Eval budget: {eval_budget}")
    print("=" * 72)

    summary: dict[str, dict] = {}

    for method_name, factory in METHODS.items():
        t0 = time.time()
        best_p = []
        best_a = []
        curves = []
        for trial_idx in range(n_trials):
            result = run_trial(
                program_name=program_name,
                agent_factory=factory,
                method_name=method_name,
                seed=trial_idx,
                eval_budget=eval_budget,
                audio_duration_s=audio_duration_s,
                loss_name=loss_name,
            )
            best_p.append(result.best_p_loss)
            best_a.append(result.best_audio_loss)
            # Best-so-far curve (P-Loss) across evals
            curves.append(np.minimum.accumulate(result.history_p_loss))
        elapsed = time.time() - t0
        summary[method_name] = {
            "p_loss_mean": float(np.mean(best_p)),
            "p_loss_std":  float(np.std(best_p)),
            "a_loss_mean": float(np.mean(best_a)),
            "elapsed_s":   elapsed,
            "curves":      curves,
        }
        print(
            f"{method_name:14s}  "
            f"P-Loss: {np.mean(best_p):.4f} ± {np.std(best_p):.4f}   "
            f"Audio: {np.mean(best_a):.4f}   "
            f"({elapsed:.1f}s)"
        )

    print("=" * 72)
    print("\nBest-so-far P-Loss at fixed eval counts (mean across trials):")
    snapshots = [10, 25, 50, 100]
    header = f"{'method':14s}  " + "  ".join(f"@{s:>3d}" for s in snapshots)
    print(header)
    for method_name, info in summary.items():
        curves = info["curves"]
        means_at = []
        for s in snapshots:
            if s <= eval_budget:
                values_at_s = [c[s - 1] for c in curves]
                means_at.append(f"{np.mean(values_at_s):6.4f}")
            else:
                means_at.append("  --  ")
        print(f"{method_name:14s}  " + "  ".join(means_at))


if __name__ == "__main__":
    main()
