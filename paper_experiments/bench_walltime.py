"""
E3 — controlled wall-clock benchmark.

Removes the confound in the recorded `duration_s` values (measured at different
times under different shared-server load) by:
  - measuring all methods INTERLEAVED (back-to-back per repeat → shared load),
  - isolating true per-eval cost from fixed setup/JIT-compile via a two-budget
    slope: per_eval = (t(B2) - t(B1)) / (B2 - B1)   [validated by the GD probe].

Also reports evals-to-threshold (from the canonical pkls) and the derived
controlled wall-clock-to-threshold = per_eval * evals_to_threshold.

Usage (keep within the 20-core cap; single process, THREADS=1):
    source experiment_scripts/env_capped.sh
    python paper_experiments/bench_walltime.py --b1 5 --b2 55 --reps 3
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.multidim import (
    CMAESAgent,
    MultiDimRandomSearch,
)
from experiments.multidim_runner import run_trial, run_trial_gd
from paper_experiments.config import SYNTH_LOSS

RES = Path(__file__).parent / "results"
SYNTHS = ["bandpass_noise", "am_noise", "add_sinesaw"]
# order chosen cheap->expensive so a partial run still has the fast methods
METHODS = ["RandomSearch", "CMA-ES", "GD"]
THRESHOLD = 0.05  # P-loss target for "time-to-solution"

_FACTORIES = {
    "RandomSearch": lambda b, s: MultiDimRandomSearch(b, seed=s),
    "CMA-ES": lambda b, s: CMAESAgent(b, sigma0=0.3, seed=s),
}


def _time_trial(synth: str, method: str, budget: int, seed: int) -> float:
    loss = SYNTH_LOSS[synth]
    t0 = time.perf_counter()
    if method == "GD":
        run_trial_gd(synth, seed=seed, eval_budget=budget, loss_name=loss)
    else:
        run_trial(synth, _FACTORIES[method], method, seed, loss_name=loss, eval_budget=budget)
    return time.perf_counter() - t0


def per_eval_ms(synth: str, method: str, b1: int, b2: int, reps: int) -> float:
    slopes = []
    for r in range(reps):
        t1 = _time_trial(synth, method, b1, r)
        t2 = _time_trial(synth, method, b2, r)
        slopes.append((t2 - t1) / (b2 - b1))
    return float(np.median(slopes)) * 1000.0


def evals_to_threshold(synth: str, method: str, thr: float) -> tuple[float, float]:
    """Returns (median evals-to-threshold over trials that reach it, reach-rate).

    The reach-rate is essential context: sec-to-threshold only averages over
    *successful* trials, so a weak method (low reach-rate) can look misleadingly
    fast — its rare successes converge quickly while most trials never reach thr.
    """
    f = RES / f"{synth}_{method}.pkl"
    if not f.exists():
        return float("nan"), float("nan")
    trials = pickle.load(open(f, "rb"))["trials"]
    hits, reached = [], []
    for t in trials:
        bsf = np.minimum.accumulate(np.array(t["history_p_loss"]))
        ok = bool(bsf[-1] <= thr)
        reached.append(ok)
        if ok:
            hits.append(int(np.argmax(bsf <= thr)) + 1)
    return (float(np.median(hits)) if hits else float("nan"),
            float(np.mean(reached)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1", type=int, default=5)
    ap.add_argument("--b2", type=int, default=55)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    print(f"E3 controlled wall-clock | two-budget slope B1={args.b1} B2={args.b2} "
          f"reps={args.reps} | threshold P-loss<={THRESHOLD}\n", flush=True)
    print(f"{'synth':14s} {'method':12s} {'ms/eval':>9s} {'reach%':>7s} {'evals→thr':>10s} {'sec→thr':>9s}", flush=True)
    print("(sec→thr is over REACHED trials only — read it together with reach%)", flush=True)
    for synth in SYNTHS:
        for method in METHODS:
            try:
                ms = per_eval_ms(synth, method, args.b1, args.b2, args.reps)
            except Exception as e:
                print(f"{synth:14s} {method:12s}   ERROR {type(e).__name__}: {e}", flush=True)
                continue
            ev, reach = evals_to_threshold(synth, method, THRESHOLD)
            sec = ms * ev / 1000.0 if ev == ev else float("nan")
            ev_s = f"{ev:.0f}" if ev == ev else "—"
            sec_s = f"{sec:.1f}" if sec == sec else "—"
            reach_s = f"{100*reach:.0f}%" if reach == reach else "—"
            print(f"{synth:14s} {method:12s} {ms:9.1f} {reach_s:>7s} {ev_s:>10s} {sec_s:>9s}", flush=True)


if __name__ == "__main__":
    main()
