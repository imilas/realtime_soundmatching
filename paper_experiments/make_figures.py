"""
Generate the figure set + a consolidated stats table for the morning analysis.

Reads the canonical pkls (+ learned_results.pkl) and writes PNGs to
paper_experiments/results/figures/ and a stats summary markdown.

Usage:
    source experiment_scripts/env_capped.sh
    python paper_experiments/make_figures.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RES = Path(__file__).parent / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SYNTHS = ["bandpass_noise", "am_noise", "add_sinesaw"]
METHODS = ["GD", "RandomSearch", "CMA-ES", "BO"]
COLORS = {
    "GD": "black", "RandomSearch": "#ff7f0e",
    "CMA-ES": "#2ca02c", "BO": "#d62728", "Learned": "#e377c2",
}
# E3 controlled wall-clock (ms/eval, reach%, sec->thr) from bench_walltime.py
E3 = {
    ("bandpass_noise", "GD"): (2212, 56, 28.8), ("bandpass_noise", "CMA-ES"): (25, 96, 1.0),
    ("bandpass_noise", "BO"): (208, 88, 8.9), ("bandpass_noise", "RandomSearch"): (22, 52, 1.5),
    ("am_noise", "GD"): (55, 14, 0.8), ("am_noise", "CMA-ES"): (286, 80, 9.4),
    ("am_noise", "BO"): (564, 91, 26.8), ("am_noise", "RandomSearch"): (283, 52, 19.8),
    ("add_sinesaw", "GD"): (24, 6, 0.3), ("add_sinesaw", "CMA-ES"): (19, 56, 0.7),
    ("add_sinesaw", "BO"): (231, 96, 13.9), ("add_sinesaw", "RandomSearch"): (19, 52, 1.3),
}


def _load(synth, method):
    f = RES / f"{synth}_{method}.pkl"
    return pickle.load(open(f, "rb"))["trials"] if f.exists() else None


def _present(synth):
    return [m for m in METHODS if _load(synth, m)]


def _save(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}", flush=True)


# 1 — best P-loss boxplots ---------------------------------------------------
def fig_boxplots():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, s in zip(axes, SYNTHS):
        ms = _present(s)
        data = [[t["best_p_loss"] for t in _load(s, m)] for m in ms]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, m in zip(bp["boxes"], ms):
            patch.set_facecolor(COLORS[m]); patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(ms) + 1)); ax.set_xticklabels(ms, rotation=35, ha="right", fontsize=8)
        ax.set_yscale("log"); ax.set_title(s, fontsize=10); ax.set_ylabel("best P-loss (log)")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Final accuracy: best P-loss per method (visited / oracle)", fontsize=12)
    fig.tight_layout(); _save(fig, "01_boxplots_bestploss.png")


# 2 — returned vs visited deception -----------------------------------------
def fig_deception():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, s in zip(axes, SYNTHS):
        ms = _present(s)
        ret, vis = [], []
        for m in ms:
            tr = _load(s, m)
            ret.append(np.median([np.array(t["history_p_loss"])[int(np.argmin(t["history_audio_loss"]))] for t in tr]))
            vis.append(np.median([np.min(t["history_p_loss"]) for t in tr]))
        x = np.arange(len(ms))
        ax.bar(x - 0.2, vis, 0.4, label="visited (oracle)", color="#aaaaaa")
        ax.bar(x + 0.2, ret, 0.4, label="returned (deployed)", color="#d62728", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(ms, rotation=35, ha="right", fontsize=8)
        ax.set_title(s, fontsize=10); ax.set_ylabel("median P-loss")
        ax.grid(True, axis="y", alpha=0.3)
        if s == SYNTHS[0]:
            ax.legend(fontsize=8)
    fig.suptitle("Deception gap: what a method VISITS vs what it RETURNS (returned − visited = deception)", fontsize=11)
    fig.tight_layout(); _save(fig, "02_returned_vs_visited.png")


# 3 — learning curves (best-so-far median + IQR) ----------------------------
def fig_learning_curves():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, s in zip(axes, SYNTHS):
        for m in _present(s):
            tr = _load(s, m)
            B = max(len(t["history_p_loss"]) for t in tr)
            curves = []
            for t in tr:
                c = np.minimum.accumulate(np.array(t["history_p_loss"]))
                if len(c) < B:
                    c = np.pad(c, (0, B - len(c)), constant_values=c[-1])
                curves.append(c)
            arr = np.vstack(curves)
            x = np.arange(1, B + 1)
            med = np.median(arr, 0)
            ax.plot(x, med, color=COLORS[m], lw=1.6, label=m)
            ax.fill_between(x, np.percentile(arr, 25, 0), np.percentile(arr, 75, 0),
                            color=COLORS[m], alpha=0.10)
        ax.set_yscale("log"); ax.set_xlabel("evaluations"); ax.set_ylabel("best-so-far P-loss (log)")
        ax.set_title(s, fontsize=10); ax.grid(True, alpha=0.3, which="both")
        if s == SYNTHS[0]:
            ax.legend(fontsize=7)
    fig.suptitle("Sample efficiency: median best-so-far P-loss (IQR band)", fontsize=12)
    fig.tight_layout(); _save(fig, "03_learning_curves.png")


# 4 — controlled wall-clock efficiency --------------------------------------
def fig_walltime():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, s in zip(axes, SYNTHS):
        ms = [m for m in METHODS if (s, m) in E3]
        msev = [E3[(s, m)][0] for m in ms]
        reach = [E3[(s, m)][1] for m in ms]
        bars = ax.bar(range(len(ms)), msev, color=[COLORS[m] for m in ms], alpha=0.8)
        ax.set_yscale("log"); ax.set_ylabel("ms / eval (log)")
        ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, rotation=35, ha="right", fontsize=8)
        ax.set_title(s, fontsize=10); ax.grid(True, axis="y", alpha=0.3)
        for b, r in zip(bars, reach):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{r}%",
                    ha="center", va="bottom", fontsize=7)
    fig.suptitle("Controlled wall-clock: ms/eval (bar) + reach-rate to P-loss≤0.05 (label)", fontsize=11)
    fig.tight_layout(); _save(fig, "04_walltime_mseval.png")


# 5 — learned vs optimizers (the headline) ----------------------------------
def fig_learned():
    p = RES / "learned_results.pkl"
    if not p.exists():
        print("  (no learned_results.pkl; skip)"); return
    L = pickle.load(open(p, "rb"))
    methods = ["Learned", "CMA-ES", "BO", "GD"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(SYNTHS)); w = 0.2
    for i, m in enumerate(methods):
        if m == "Learned":
            vals = [L[s]["median"] for s in SYNTHS]
        else:
            key = {"CMA-ES": "cma", "BO": "bo", "GD": "gd"}[m]
            vals = [L[s][key] for s in SYNTHS]
        ax.bar(x + (i - 1.5) * w, vals, w, label=m + (" (0-eval)" if m == "Learned" else " (200)"),
               color=COLORS[m], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(SYNTHS); ax.set_ylabel("returned P-loss (median)")
    ax.set_title("Learned amortized model (zero-shot) vs optimizers (200 evals)\n"
                 "learned WINS on the non-identifiable add_sinesaw", fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); _save(fig, "05_learned_vs_optimizers.png")


# 6 — non-identifiability scatter -------------------------------------------
def fig_identifiability():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, s in zip(axes, SYNTHS):
        tr = _load(s, "CMA-ES")
        A, P = [], []
        for t in tr[:60]:
            A.extend(t["history_audio_loss"]); P.extend(t["history_p_loss"])
        A, P = np.array(A), np.array(P)
        ax.scatter(A, P, s=4, alpha=0.25, color="#2ca02c")
        corr = np.corrcoef(A, P)[0, 1]
        ax.set_title(f"{s}\ncorr(audio, P-loss)={corr:.2f}", fontsize=9)
        ax.set_xlabel("audio loss"); ax.set_ylabel("P-loss")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Loss informativeness: audio-loss vs P-loss (flat/low-corr ⇒ non-identifiable)", fontsize=11)
    fig.tight_layout(); _save(fig, "06_identifiability_scatter.png")


def _med_returned(tr):
    return np.median([np.array(t["history_p_loss"])[int(np.argmin(t["history_audio_loss"]))] for t in tr])


def _med_visited(tr):
    return np.median([np.min(t["history_p_loss"]) for t in tr])


def stats_summary():
    """Write a consolidated markdown stats reference."""
    L = pickle.load(open(RES / "learned_results.pkl", "rb")) if (RES / "learned_results.pkl").exists() else None
    out = ["# Stats summary (auto-generated 2026-05-31)\n",
           "All P-loss = Euclidean distance in normalized param space (lower=better),"
           " budget 200, matched seeds; Learned is zero-shot (0 evals).\n"]

    def table(title, fn, with_learned=True, learned_key="median"):
        hdr = METHODS + (["Learned"] if with_learned else [])
        rows = [f"## {title}\n", "| synth | " + " | ".join(hdr) + " |",
                "|" + "---|" * (len(hdr) + 1)]
        for s in SYNTHS:
            r = [s]
            for m in METHODS:
                tr = _load(s, m)
                r.append(f"{fn(tr):.3f}" if tr else "—")
            if with_learned:
                r.append(f"{L[s][learned_key]:.3f}" if L else "—")
            rows.append("| " + " | ".join(r) + " |")
        return rows + [""]

    out += table("Returned P-loss (median) — what each method DEPLOYS", _med_returned)
    out += table("Visited P-loss (median) — oracle best (optimistic)", _med_visited)

    # deception gap
    out += ["## Deception gap (returned − visited; higher = more fooled by the loss)\n",
            "| synth | " + " | ".join(METHODS) + " |", "|" + "---|" * (len(METHODS) + 1)]
    for s in SYNTHS:
        r = [s]
        for m in METHODS:
            tr = _load(s, m)
            r.append(f"{_med_returned(tr) - _med_visited(tr):.3f}" if tr else "—")
        out.append("| " + " | ".join(r) + " |")
    out.append("")

    # wall-clock
    out += ["## Controlled wall-clock (E3): ms/eval | reach% (P≤0.05) | sec→thr\n",
            "| synth | " + " | ".join(METHODS) + " |", "|" + "---|" * (len(METHODS) + 1)]
    for s in SYNTHS:
        r = [s]
        for m in METHODS:
            if (s, m) in E3:
                ms, rc, sec = E3[(s, m)]
                r.append(f"{ms} / {rc}% / {sec}s")
            else:
                r.append("—")
        out.append("| " + " | ".join(r) + " |")
    out += ["",
            "## Headline takeaways\n",
            "- **CMA-ES** best on identifiable synths (bandpass 0.001, am_noise 0.008), gap≈0.",
            "- **BO** least-deceived optimizer on flat add_sinesaw (returned 0.197) — global hedging.",
            "- **GD** weakest + most deceived (am 0.488, add 0.520) and ~29× slower wall-clock on bandpass.",
            "- **QL** does not learn (flat across thousands of trials).",
            "- **Learned (0-eval)** beats BO/GD everywhere and **beats ALL optimizers ~4× on the "
            "non-identifiable add_sinesaw (0.045 vs 0.197)** — amortized priors win where the loss is useless.",
            ""]
    p = RES / "STATS_SUMMARY.md"
    p.write_text("\n".join(out) + "\n")
    print(f"  wrote {p}", flush=True)


if __name__ == "__main__":
    print("Generating figures ->", FIG, flush=True)
    fig_boxplots()
    fig_deception()
    fig_learning_curves()
    fig_walltime()
    fig_learned()
    fig_identifiability()
    stats_summary()
    print("Done.", flush=True)
