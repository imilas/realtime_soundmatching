"""
Generate the figure set + a consolidated stats table for the morning analysis.

Reads the canonical pkls and writes PNGs to
paper_experiments/results/figures/ and a stats summary markdown.

Usage:
    source experiment_scripts/env_capped.sh
    python paper_experiments/make_figures.py
"""
from __future__ import annotations

import pickle
import re
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

SYNTHS = ["bandpass_noise_v1", "am_noise", "add_sinesaw", "sine_mod_saw", "chirplet",
          "dx7_alg1", "dx7_alg2", "dx7_alg3"]
LOSSES = ["SIMSE_Spec", "DTW_Envelope", "JTFS", "L1_Spec", "CLAP"]
METHODS = ["GD", "RandomSearch", "CMA-ES", "LES"]
COLORS = {
    "GD": "black", "RandomSearch": "#ff7f0e",
    "CMA-ES": "#2ca02c", "LES": "#9467bd",
}

# canonical loss per synth, used for figures that show one panel per synth
SYNTH_LOSS_CANONICAL = {
    "bandpass_noise_v1": "SIMSE_Spec",
    "am_noise": "DTW_Envelope",
    "add_sinesaw": "SIMSE_Spec",
    "sine_mod_saw": "JTFS",
    "chirplet": "JTFS",
    "dx7_alg1": "SIMSE_Spec",
    "dx7_alg2": "SIMSE_Spec",
    "dx7_alg3": "SIMSE_Spec",
}

# E3 controlled wall-clock (ms/eval, reach%, sec->thr) from bench_walltime.py
# only measured for the original 3 synths at their canonical loss.
E3 = {
    ("bandpass_noise_v1", "GD"): (2212, 56, 28.8), ("bandpass_noise_v1", "CMA-ES"): (25, 96, 1.0),
    ("bandpass_noise_v1", "RandomSearch"): (22, 52, 1.5),
    ("am_noise", "GD"): (55, 14, 0.8), ("am_noise", "CMA-ES"): (286, 80, 9.4),
    ("am_noise", "RandomSearch"): (283, 52, 19.8),
    ("add_sinesaw", "GD"): (24, 6, 0.3), ("add_sinesaw", "CMA-ES"): (19, 56, 0.7),
    ("add_sinesaw", "RandomSearch"): (19, 52, 1.3),
}
E3_SYNTHS = ["bandpass_noise_v1", "am_noise", "add_sinesaw"]


def _slug(loss):
    return re.sub(r"[^A-Za-z0-9]+", "_", loss).strip("_")


def _load(synth, loss, method):
    f = RES / f"{synth}_{_slug(loss)}_{method}.pkl"
    return pickle.load(open(f, "rb"))["trials"] if f.exists() else None


def _present(synth, loss):
    return [m for m in METHODS if _load(synth, loss, m)]


def _save(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}", flush=True)


# 1 — best P-loss boxplots ---------------------------------------------------
def fig_boxplots():
    fig, axes = plt.subplots(len(LOSSES), len(SYNTHS), figsize=(4 * len(SYNTHS), 3.2 * len(LOSSES)))
    for i, loss in enumerate(LOSSES):
        for j, s in enumerate(SYNTHS):
            ax = axes[i, j]
            ms = _present(s, loss)
            if not ms:
                ax.axis("off")
                continue
            data = [[t["best_p_loss"] for t in _load(s, loss, m)] for m in ms]
            bp = ax.boxplot(data, patch_artist=True, showfliers=False)
            for patch, m in zip(bp["boxes"], ms):
                patch.set_facecolor(COLORS[m]); patch.set_alpha(0.6)
            ax.set_xticks(range(1, len(ms) + 1)); ax.set_xticklabels(ms, rotation=35, ha="right", fontsize=7)
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(s, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{loss}\nbest P-loss (log)", fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Final accuracy: best P-loss per method (visited / oracle), rows = loss, cols = synth", fontsize=12)
    fig.tight_layout(); _save(fig, "01_boxplots_bestploss.png")


# 2 — returned vs visited deception -----------------------------------------
def fig_deception():
    fig, axes = plt.subplots(len(LOSSES), len(SYNTHS), figsize=(4 * len(SYNTHS), 3.2 * len(LOSSES)))
    for i, loss in enumerate(LOSSES):
        for j, s in enumerate(SYNTHS):
            ax = axes[i, j]
            ms = _present(s, loss)
            if not ms:
                ax.axis("off")
                continue
            ret, vis = [], []
            for m in ms:
                tr = _load(s, loss, m)
                ret.append(np.median([t["history_p_loss"][-1] for t in tr]))
                vis.append(np.median([np.min(t["history_p_loss"]) for t in tr]))
            x = np.arange(len(ms))
            ax.bar(x - 0.2, vis, 0.4, label="visited (oracle)", color="#aaaaaa")
            ax.bar(x + 0.2, ret, 0.4, label="returned final", color="#d62728", alpha=0.8)
            ax.set_xticks(x); ax.set_xticklabels(ms, rotation=35, ha="right", fontsize=7)
            if i == 0:
                ax.set_title(s, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{loss}\nmedian P-loss", fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Final P-loss vs best visited: what a method ENDS AT vs best it SAW (final − visited = gap)", fontsize=11)
    fig.tight_layout(); _save(fig, "02_returned_vs_visited.png")


# 3 — learning curves (best-so-far median + IQR) ----------------------------
def fig_learning_curves():
    fig, axes = plt.subplots(len(LOSSES), len(SYNTHS), figsize=(4 * len(SYNTHS), 3.2 * len(LOSSES)))
    for i, loss in enumerate(LOSSES):
        for j, s in enumerate(SYNTHS):
            ax = axes[i, j]
            ms = _present(s, loss)
            if not ms:
                ax.axis("off")
                continue
            for m in ms:
                tr = _load(s, loss, m)
                B = max(len(t["history_p_loss"]) for t in tr)
                curves = []
                for t in tr:
                    # fmin.accumulate ignores occasional NaN p-loss evaluations
                    # (transient eval failures) instead of poisoning the running
                    # best from that point on.
                    c = np.fmin.accumulate(np.array(t["history_p_loss"]))
                    if len(c) < B:
                        c = np.pad(c, (0, B - len(c)), constant_values=c[-1])
                    curves.append(c)
                arr = np.vstack(curves)
                x = np.arange(1, B + 1)
                med = np.nanmedian(arr, 0)
                ax.plot(x, med, color=COLORS[m], lw=1.6, label=m)
                ax.fill_between(x, np.nanpercentile(arr, 25, 0), np.nanpercentile(arr, 75, 0),
                                color=COLORS[m], alpha=0.10)
            ax.set_yscale("log")
            if i == len(LOSSES) - 1:
                ax.set_xlabel("evaluations")
            if j == 0:
                ax.set_ylabel(f"{loss}\nbest-so-far P-loss (log)", fontsize=8)
            if i == 0:
                ax.set_title(s, fontsize=10)
            ax.grid(True, alpha=0.3, which="both")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Sample efficiency: median best-so-far P-loss (IQR band), rows = loss, cols = synth", fontsize=12)
    fig.tight_layout(); _save(fig, "03_learning_curves.png")


# 3b — returned (instantaneous) P-loss, no running-min ----------------------
def fig_returned_curves():
    """Same layout as fig_learning_curves, but plots the *raw* per-step
    P-loss (no cumulative minimum). A method whose best-so-far curve looks
    good only because it occasionally wanders near a good point — without
    settling there — will show a flat/noisy median here instead of a
    monotonically improving one."""
    fig, axes = plt.subplots(len(LOSSES), len(SYNTHS), figsize=(4 * len(SYNTHS), 3.2 * len(LOSSES)))
    for i, loss in enumerate(LOSSES):
        for j, s in enumerate(SYNTHS):
            ax = axes[i, j]
            ms = _present(s, loss)
            if not ms:
                ax.axis("off")
                continue
            for m in ms:
                tr = _load(s, loss, m)
                B = max(len(t["history_p_loss"]) for t in tr)
                curves = []
                for t in tr:
                    c = np.array(t["history_p_loss"], dtype=float)
                    if len(c) < B:
                        c = np.pad(c, (0, B - len(c)), constant_values=np.nan)
                    curves.append(c)
                arr = np.vstack(curves)
                x = np.arange(1, B + 1)
                med = np.nanmedian(arr, 0)
                ax.plot(x, med, color=COLORS[m], lw=1.6, label=m)
                ax.fill_between(x, np.nanpercentile(arr, 25, 0), np.nanpercentile(arr, 75, 0),
                                color=COLORS[m], alpha=0.10)
            if i == len(LOSSES) - 1:
                ax.set_xlabel("evaluations")
            if j == 0:
                ax.set_ylabel(f"{loss}\nreturned P-loss", fontsize=8)
            if i == 0:
                ax.set_title(s, fontsize=10)
            ax.grid(True, alpha=0.3, which="both")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Sanity check: median returned (instantaneous, non-cumulative) P-loss (IQR band), "
                  "rows = loss, cols = synth", fontsize=12)
    fig.tight_layout(); _save(fig, "07_returned_curves.png")


# 4 — controlled wall-clock efficiency --------------------------------------
def fig_walltime():
    fig, axes = plt.subplots(1, len(E3_SYNTHS), figsize=(4 * len(E3_SYNTHS), 4))
    for ax, s in zip(axes, E3_SYNTHS):
        ms = [m for m in METHODS if (s, m) in E3]
        msev = [E3[(s, m)][0] for m in ms]
        reach = [E3[(s, m)][1] for m in ms]
        bars = ax.bar(range(len(ms)), msev, color=[COLORS[m] for m in ms], alpha=0.8)
        ax.set_yscale("log"); ax.set_ylabel("ms / eval (log)")
        ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{s}\n({SYNTH_LOSS_CANONICAL[s]})", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        for b, r in zip(bars, reach):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{r}%",
                    ha="center", va="bottom", fontsize=7)
    fig.suptitle("Controlled wall-clock: ms/eval (bar) + reach-rate to P-loss≤0.05 (label)\n"
                  "(measured for these 3 synths only)", fontsize=11)
    fig.tight_layout(); _save(fig, "04_walltime_mseval.png")


# 5 — CLAP wall-clock (from production run durations) -----------------------
def fig_walltime_clap():
    """ms/eval for the CLAP loss, derived from each trial's recorded
    `duration_s` / `eval_budget` — no controlled E3 benchmark exists for CLAP
    (it wasn't part of bench_walltime.py), so this uses the actual production
    run timings instead."""
    clap_methods = [m for m in METHODS if m != "GD"]  # GD doesn't support CLAP
    fig, axes = plt.subplots(1, len(SYNTHS), figsize=(3.2 * len(SYNTHS), 4))
    for ax, s in zip(axes, SYNTHS):
        ms, msev = [], []
        for m in clap_methods:
            tr = _load(s, "CLAP", m)
            if not tr:
                continue
            budget = tr[0].get("eval_budget", len(tr[0].get("history_p_loss", [])))
            durs = [t["duration_s"] for t in tr if t.get("duration_s") is not None]
            if not durs or not budget:
                continue
            ms.append(m)
            msev.append(np.median(durs) / budget * 1000)
        if not ms:
            ax.axis("off")
            continue
        ax.bar(range(len(ms)), msev, color=[COLORS[m] for m in ms], alpha=0.8)
        ax.set_yscale("log"); ax.set_ylabel("ms / eval (log)")
        ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, rotation=35, ha="right", fontsize=8)
        ax.set_title(s, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("CLAP wall-clock: ms/eval from production run durations (median duration_s / eval_budget)",
                  fontsize=11)
    fig.tight_layout(); _save(fig, "05_clap_walltime.png")


# 6 — non-identifiability scatter -------------------------------------------
def fig_identifiability():
    fig, axes = plt.subplots(len(LOSSES), len(SYNTHS), figsize=(4 * len(SYNTHS), 3.2 * len(LOSSES)))
    for i, loss in enumerate(LOSSES):
        for j, s in enumerate(SYNTHS):
            ax = axes[i, j]
            ms = _present(s, loss)
            ref = "CMA-ES" if "CMA-ES" in ms else (ms[0] if ms else None)
            if ref is None:
                ax.axis("off")
                continue
            tr = _load(s, loss, ref)
            A, P = [], []
            for t in tr[:60]:
                A.extend(t["history_audio_loss"]); P.extend(t["history_p_loss"])
            A, P = np.array(A), np.array(P)
            ax.scatter(A, P, s=4, alpha=0.25, color="#2ca02c")
            corr = np.corrcoef(A, P)[0, 1]
            if i == 0:
                ax.set_title(f"{s}\ncorr={corr:.2f}", fontsize=9)
            else:
                ax.set_title(f"corr={corr:.2f}", fontsize=9)
            if i == len(LOSSES) - 1:
                ax.set_xlabel("audio loss")
            if j == 0:
                ax.set_ylabel(f"{loss}\nP-loss", fontsize=8)
            ax.grid(True, alpha=0.3)
    fig.suptitle("Loss informativeness: audio-loss vs P-loss (flat/low-corr ⇒ non-identifiable), "
                  "rows = loss, cols = synth", fontsize=11)
    fig.tight_layout(); _save(fig, "06_identifiability_scatter.png")


def _med_returned(tr):
    return np.median([t["history_p_loss"][-1] for t in tr])


def _med_visited(tr):
    return np.median([np.min(t["history_p_loss"]) for t in tr])


def stats_summary():
    """Write a consolidated markdown stats reference (canonical loss per synth)."""
    out = ["# Stats summary (auto-generated)\n",
           "All P-loss = Euclidean distance in normalized param space (lower=better),"
           " budget 200, matched seeds."
           " Each synth uses its canonical loss (see SYNTH_LOSS_CANONICAL).\n"]

    def table(title, fn):
        hdr = METHODS
        rows = [f"## {title}\n", "| synth | " + " | ".join(hdr) + " |",
                "|" + "---|" * (len(hdr) + 1)]
        for s in SYNTHS:
            loss = SYNTH_LOSS_CANONICAL[s]
            r = [s]
            for m in METHODS:
                tr = _load(s, loss, m)
                r.append(f"{fn(tr):.3f}" if tr else "—")
            rows.append("| " + " | ".join(r) + " |")
        return rows + [""]

    out += table("Final P-loss (median) — last evaluation of each trial", _med_returned)
    out += table("Visited P-loss (median) — oracle best (optimistic)", _med_visited)

    # deception gap
    out += ["## Final vs visited gap (final − visited; higher = final worse than best seen)\n",
            "| synth | " + " | ".join(METHODS) + " |", "|" + "---|" * (len(METHODS) + 1)]
    for s in SYNTHS:
        loss = SYNTH_LOSS_CANONICAL[s]
        r = [s]
        for m in METHODS:
            tr = _load(s, loss, m)
            r.append(f"{_med_returned(tr) - _med_visited(tr):.3f}" if tr else "—")
        out.append("| " + " | ".join(r) + " |")
    out.append("")

    # wall-clock
    out += ["## Controlled wall-clock (E3): ms/eval | reach% (P≤0.05) | sec→thr\n",
            "| synth | " + " | ".join(METHODS) + " |", "|" + "---|" * (len(METHODS) + 1)]
    for s in E3_SYNTHS:
        r = [s]
        for m in METHODS:
            if (s, m) in E3:
                ms, rc, sec = E3[(s, m)]
                r.append(f"{ms} / {rc}% / {sec}s")
            else:
                r.append("—")
        out.append("| " + " | ".join(r) + " |")
    out.append("")

    p = RES / "STATS_SUMMARY.md"
    p.write_text("\n".join(out) + "\n")
    print(f"  wrote {p}", flush=True)


if __name__ == "__main__":
    print("Generating figures ->", FIG, flush=True)
    fig_boxplots()
    fig_deception()
    fig_learning_curves()
    fig_returned_curves()
    fig_walltime()
    fig_walltime_clap()
    fig_identifiability()
    stats_summary()
    print("Done.", flush=True)
