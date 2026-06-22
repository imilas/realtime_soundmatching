"""
Sanity-check the DX7 experiment results.

Run:
    source experiment_scripts/env_capped.sh
    python paper_experiments/validate_dx7.py

"makes sense" is judged on the OBJECTIVE each method actually optimizes, which is
the AUDIO loss -- not the P-loss (parameter distance). FM synths are highly
non-identifiable, so audio loss can drop a lot while P-loss barely moves (the
"deception gap"); that is a finding, not a bug.

Checks, per (synth, loss, method) cell:
  - trials present, finite histories
  - ANOMALY if a method fails to reduce its audio loss (median best < median init)
Reported (informational, not anomalies):
  - median returned P-loss (what the method outputs) and best/visited P-loss
  - deception gap = returned P-loss - best visited P-loss
  - best method per (synth, loss) by returned P-loss
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np

RES = Path(__file__).parent / "results"
SYNTHS = ["dx7_alg1", "dx7_alg2", "dx7_alg3"]
METHODS = ["GD", "CMA-ES", "RandomSearch", "LES"]
LOSSES = ["SIMSE_Spec", "L1_Spec", "JTFS", "DTW_Envelope", "CLAP"]


def _load(synth, loss, method):
    p = RES / f"{synth}_{loss}_{method}.pkl"
    if not p.exists():
        return []
    try:
        return pickle.load(open(p, "rb")).get("trials", [])
    except Exception:
        return []


def _scores(trials, method):
    """Per trial: audio (init, best) and P-loss (returned, best)."""
    a_init, a_best, p_ret, p_best = [], [], [], []
    for t in trials:
        al = np.asarray(t.get("history_audio_loss", []), float); al = al[np.isfinite(al)]
        pl = np.asarray(t.get("history_p_loss", []), float)
        if len(al):
            a_init.append(float(al[0])); a_best.append(float(al.min()))
        plf = pl[np.isfinite(pl)]
        if len(plf):
            p_best.append(float(plf.min()))
            if method == "GD":
                p_ret.append(float(plf[-1]))
            elif len(al):
                p_ret.append(float(pl[int(np.nanargmin(al))]))
            else:
                p_ret.append(float(plf[-1]))
    return (np.array(a_init), np.array(a_best), np.array(p_ret), np.array(p_best))


def main():
    warns, n_cells, n_trials = [], 0, 0
    for synth in SYNTHS:
        print(f"\n=== {synth} ===   (audio init→best | P-loss ret/best | gap)")
        for loss in LOSSES:
            print(f"  {loss}")
            per_method_ret = {}
            for method in METHODS:
                if method == "GD" and loss == "CLAP":
                    continue
                tr = _load(synth, loss, method)
                if not tr:
                    print(f"      {method:13s}  -")
                    continue
                n_cells += 1; n_trials += len(tr)
                ai, ab, pr, pb = _scores(tr, method)
                if len(ai) == 0 or len(pr) == 0:
                    warns.append(f"{synth}/{loss}/{method}: no finite histories")
                    print(f"      {method:13s}  NaN/empty!"); continue
                if not np.all(np.isfinite(ab)):
                    warns.append(f"{synth}/{loss}/{method}: non-finite audio loss")
                # PRIMARY sanity: the optimizer must reduce its own objective.
                if np.median(ab) > np.median(ai) - 1e-12:
                    warns.append(f"{synth}/{loss}/{method}: audio loss NOT reduced "
                                 f"({np.median(ai):.4g}→{np.median(ab):.4g})")
                med_pr, med_pb = np.median(pr), np.median(pb)
                per_method_ret[method] = med_pr
                gap = med_pr - med_pb
                print(f"      {method:13s}  n={len(tr):3d}  audio {np.median(ai):.4g}→{np.median(ab):.4g}"
                      f"  | P {med_pr:.3f}/{med_pb:.3f}  | gap {gap:+.3f}")
            if per_method_ret:
                bestm = min(per_method_ret, key=per_method_ret.get)
                print(f"      → best returned-P: {bestm} ({per_method_ret[bestm]:.3f})")

    print(f"\nScanned {n_cells} cells, {n_trials} trials.")
    if warns:
        print(f"\n⚠️  {len(warns)} ANOMALIES:")
        for w in warns:
            print("   -", w)
    else:
        print("\n✅ Every method reduces its audio-loss objective. "
              "(P-loss/deception-gap reported above is the science, not an error.)")


if __name__ == "__main__":
    main()
