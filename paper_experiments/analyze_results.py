"""
Diagnostic analysis of paper_experiments/results/*.pkl.

Regenerates every number in ANALYSIS.md:
  - step-size distribution per method (tests "BO behaves like random search")
  - final accuracy (best_p_loss) per (synth, method)
  - sample efficiency (best-so-far at eval snapshots)
  - BO best-found-index (does the GP phase help over initial random?)
  - add_sinesaw identifiability (audio-loss vs P-loss correlation)
  - QL learning (first-20% vs last-20% of trials)

Usage:
    python paper_experiments/analyze_results.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.params import FaustParams
from synths.build import prepare

RES = Path(__file__).parent / "results"
SYNTHS = ["bandpass_noise", "am_noise", "add_sinesaw"]
METHODS = ["GD", "RandomSearch", "CMA-ES", "BO"]
SNAP = [25, 50, 100, 200]


def _bounds(synth: str):
    b = prepare(synth)
    p = FaustParams(str(b.json_path))
    lo, hi = p.bounds()
    return np.array(lo, float), np.array(hi, float), p.names()


def _norm_traj(bounds, hist):
    lo, hi, names = bounds
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    return np.array(
        [[(float(hp[n]) - lo[i]) / rng[i] for i, n in enumerate(names)] for hp in hist]
    )


def _load(synth, method):
    f = RES / f"{synth}_{method}.pkl"
    return pickle.load(open(f, "rb"))["trials"] if f.exists() else None


def step_sizes(sample=30):
    print("\n=== STEP SIZES (normalized, ||x_t - x_{t-1}||) ===")
    print(f"{'synth':14s} {'method':12s} {'mean':>7s} {'median':>7s} {'p90':>7s} {'max':>7s}")
    for s in SYNTHS:
        b = _bounds(s)
        for m in METHODS:
            trials = _load(s, m)
            if not trials:
                continue
            steps = []
            for t in trials[:sample]:
                tr = _norm_traj(b, t["history_params"])
                if len(tr) >= 2:
                    steps.extend(np.linalg.norm(np.diff(tr, axis=0), axis=1))
            steps = np.array(steps)
            if len(steps):
                print(f"{s:14s} {m:12s} {steps.mean():7.3f} {np.median(steps):7.3f} "
                      f"{np.percentile(steps,90):7.3f} {steps.max():7.3f}")


def final_accuracy():
    print("\n=== FINAL ACCURACY best_p_loss ===")
    print(f"{'synth':14s} {'method':12s} {'n':>5s} {'mean':>7s} {'median':>7s} {'std':>7s} {'min':>7s}")
    for s in SYNTHS:
        for m in METHODS:
            trials = _load(s, m)
            if not trials:
                continue
            v = np.array([t["best_p_loss"] for t in trials])
            print(f"{s:14s} {m:12s} {len(v):5d} {v.mean():7.3f} {np.median(v):7.3f} "
                  f"{v.std():7.3f} {v.min():7.3f}")


def sample_efficiency():
    print("\n=== BEST-SO-FAR at eval snapshots (median) ===")
    print(f"{'synth':14s} {'method':12s} " + " ".join(f"@{k:>4d}" for k in SNAP))
    for s in SYNTHS:
        for m in METHODS:
            trials = _load(s, m)
            if not trials:
                continue
            bs = [np.minimum.accumulate(np.array(t["history_p_loss"]))
                  for t in trials if len(t["history_p_loss"])]
            L = max(len(x) for x in bs)
            arr = np.array([np.pad(x, (0, L - len(x)), constant_values=x[-1]) for x in bs])
            med = np.median(arr, 0)
            print(f"{s:14s} {m:12s} " + " ".join(f"{med[min(k,L)-1]:5.3f}" for k in SNAP))


def bo_best_index():
    print("\n=== BO best-found index (n_initial=10) ===")
    print(f"{'synth':14s} {'median_idx':>10s} {'%first10':>9s} {'%after100':>10s}")
    for s in SYNTHS:
        trials = _load(s, "BO")
        if not trials:
            continue
        idx = np.array([int(np.argmin(t["history_p_loss"])) for t in trials])
        print(f"{s:14s} {np.median(idx):10.0f} {100*np.mean(idx<10):8.0f}% {100*np.mean(idx>=100):9.0f}%")


def identifiability():
    print("\n=== add_sinesaw identifiability (audio-loss vs P-loss) ===")
    for m in ["CMA-ES", "BO", "HillClimber"]:
        trials = _load("add_sinesaw", m)
        if not trials:
            continue
        A, P = [], []
        for t in trials[:50]:
            A.extend(t["history_audio_loss"])
            P.extend(t["history_p_loss"])
        corr = np.corrcoef(A, P)[0, 1]
        ploss_at_min_audio = [np.array(t["history_p_loss"])[int(np.argmin(t["history_audio_loss"]))]
                              for t in trials]
        print(f"  {m:12s} corr(audio,P)={corr:.3f}  "
              f"median P at min-audio={np.median(ploss_at_min_audio):.3f}")


def returned_vs_visited():
    """RETURNED (argmin audio-loss) vs VISITED (oracle best) P-loss — the gap
    measures how badly the loss deceives each method (key comparative metric)."""
    print("\n=== RETURNED vs VISITED P-loss (median); gap = deception ===")
    print(f"{'synth':14s} {'method':12s} {'returned':>9s} {'visited':>8s} {'gap':>6s}")
    for s in SYNTHS:
        for m in METHODS:
            trials = _load(s, m)
            if not trials:
                continue
            ret, vis = [], []
            for t in trials:
                ha = np.array(t["history_audio_loss"])
                hp = np.array(t["history_p_loss"])
                if len(ha) == 0:
                    continue
                ret.append(hp[int(np.argmin(ha))])
                vis.append(hp.min())
            r, v = np.median(ret), np.median(vis)
            print(f"{s:14s} {m:12s} {r:9.3f} {v:8.3f} {r-v:6.3f}")


if __name__ == "__main__":
    final_accuracy()
    returned_vs_visited()
    sample_efficiency()
    step_sizes()
    bo_best_index()
    identifiability()
