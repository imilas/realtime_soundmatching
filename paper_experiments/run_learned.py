"""
E5 B1 — train the amortized inverse model and evaluate it ZERO-SHOT against the
classical optimizers on the exact benchmark targets (seeds 0-199).

The learned model returns its parameter prediction directly (0 audio
evaluations), so its "returned P-loss" = ||pred_norm - true_norm|| is directly
comparable to the optimizers' returned P-loss (argmin audio-loss) at 200 evals.

Usage:
    source experiment_scripts/env_capped.sh
    python paper_experiments/run_learned.py --n-train 10000
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.learned.data_gen import generate_dataset, benchmark_targets
from paper_experiments.config import SYNTHS

RES = Path(__file__).parent / "results"
OUT = RES / "learned_results.pkl"


def _optimizer_returned(synth: str, method: str) -> float:
    """Median returned P-loss (argmin audio-loss) for an optimizer, for context."""
    f = RES / f"{synth}_{method}.pkl"
    if not f.exists():
        return float("nan")
    trials = pickle.load(open(f, "rb"))["trials"]
    vals = [np.array(t["history_p_loss"])[int(np.argmin(t["history_audio_loss"]))] for t in trials]
    return float(np.median(vals))


def run(n_train: int, seed: int) -> None:
    from sklearn.neural_network import MLPRegressor

    results = {}
    print(f"E5 B1 amortized inverse model | n_train={n_train}\n", flush=True)
    print(f"{'synth':14s} {'learned(0-eval)':>16s} {'CMA-ES':>8s} {'BO':>8s} {'GD':>8s} {'reach%':>7s}", flush=True)
    for synth in SYNTHS:
        Xtr, Ytr = generate_dataset(synth, n_train, seed=seed)
        model = MLPRegressor(
            hidden_layer_sizes=(256, 128), activation="relu",
            max_iter=300, early_stopping=True, random_state=seed,
        )
        model.fit(Xtr, Ytr)

        Xte, Yte = benchmark_targets(synth, n_seeds=200)
        pred = np.clip(model.predict(Xte), 0.0, 1.0)
        ploss = np.linalg.norm(pred - Yte, axis=1)  # returned P-loss, 0 evals
        med = float(np.median(ploss))
        reach = float(np.mean(ploss <= 0.05))

        cma = _optimizer_returned(synth, "CMA-ES")
        bo = _optimizer_returned(synth, "BO")
        gd = _optimizer_returned(synth, "GD")
        print(f"{synth:14s} {med:16.3f} {cma:8.3f} {bo:8.3f} {gd:8.3f} {100*reach:6.0f}%", flush=True)
        results[synth] = dict(ploss=ploss, median=med, reach=reach,
                              cma=cma, bo=bo, gd=gd, pred=pred, true=Yte)

    with open(OUT, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved {OUT}", flush=True)
    print("Note: learned column is ZERO-SHOT (0 audio evals); optimizers used 200.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.n_train, args.seed)
