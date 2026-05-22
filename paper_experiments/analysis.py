"""
Analysis pipeline for the paper experiment results.

Reads paper_experiments/results/results.csv and produces:
  - Bootstrap 95% CI on best P-Loss per (synth, method)
  - NPSK ranking matrix (fraction of trials where method A beats B)
  - Learning curves (best-so-far P-Loss vs eval count)
  - Saved plots in paper_experiments/results/figures/

Usage:
    python paper_experiments/analysis.py
    python paper_experiments/analysis.py --csv path/to/results.csv
    python paper_experiments/analysis.py --snapshots 50 100 200 400 --no-plots
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_CSV  = Path(__file__).parent / "results" / "results.csv"
FIGURES_DIR  = Path(__file__).parent / "results" / "figures"

METHOD_ORDER = ["GD", "HillClimber", "RandomSearch", "CMA-ES", "BO"]
SYNTH_ORDER  = ["bandpass_noise", "sine_saw", "am_noise", "sine_mod_saw", "sine_mod_sine"]
N_BOOTSTRAP  = 1000
SNAPSHOTS    = [50, 100, 200, 400]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["history_p_loss"]     = df["history_p_loss"].apply(json.loads)
    df["history_audio_loss"] = df["history_audio_loss"].apply(json.loads)
    return df


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(values: np.ndarray, n: int = N_BOOTSTRAP, alpha: float = 0.05) -> tuple[float, float]:
    """Return (lower, upper) bootstrap percentile CI for the mean."""
    rng = np.random.default_rng(0)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n)]
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean best P-Loss ± bootstrap CI per (synth, method)."""
    rows = []
    for synth in SYNTH_ORDER:
        for method in METHOD_ORDER:
            sub = df[(df["program"] == synth) & (df["method"] == method)]["best_p_loss"].values
            if len(sub) == 0:
                continue
            lo, hi = bootstrap_ci(sub)
            rows.append({
                "synth":  synth,
                "method": method,
                "n":      len(sub),
                "mean":   sub.mean(),
                "std":    sub.std(),
                "ci_lo":  lo,
                "ci_hi":  hi,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# NPSK ranking
# ---------------------------------------------------------------------------

def npsk_matrix(df: pd.DataFrame, synth: str) -> pd.DataFrame:
    """
    NPSK pairwise win-rate matrix for one synth.

    Entry (A, B) = fraction of (trial_A, trial_B) pairs where A has lower
    best P-Loss than B. Matched by seed when possible; unmatched otherwise.
    """
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    mat = pd.DataFrame(index=methods, columns=methods, dtype=float)
    sub = df[df["program"] == synth]

    for a, b in combinations(methods, 2):
        vals_a = sub[sub["method"] == a].set_index("seed")["best_p_loss"]
        vals_b = sub[sub["method"] == b].set_index("seed")["best_p_loss"]
        shared = vals_a.index.intersection(vals_b.index)
        if len(shared) == 0:
            mat.loc[a, b] = np.nan
            mat.loc[b, a] = np.nan
            continue
        a_wins = (vals_a[shared].values < vals_b[shared].values).mean()
        mat.loc[a, b] = a_wins
        mat.loc[b, a] = 1.0 - a_wins

    np.fill_diagonal(mat.values, 0.5)
    return mat


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------

def mean_best_so_far(df: pd.DataFrame, synth: str, method: str, max_evals: int) -> np.ndarray:
    """Mean best-so-far P-Loss curve across trials, shape (max_evals,)."""
    rows = df[(df["program"] == synth) & (df["method"] == method)]
    curves = []
    for hist in rows["history_p_loss"]:
        h = np.array(hist[:max_evals])
        curves.append(np.minimum.accumulate(h))
    if not curves:
        return np.full(max_evals, np.nan)
    # Pad shorter histories with their last value.
    padded = np.array([
        np.pad(c, (0, max_evals - len(c)), constant_values=c[-1]) if len(c) < max_evals else c
        for c in curves
    ])
    return padded.mean(axis=0)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    tbl = summary_table(df)
    for synth in SYNTH_ORDER:
        sub = tbl[tbl["synth"] == synth]
        if sub.empty:
            continue
        print(f"\n{synth}")
        print(f"  {'method':14s}  {'n':>4s}  {'mean':>7s}  {'std':>7s}  {'95% CI':>16s}")
        for _, row in sub.iterrows():
            ci = f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]"
            print(f"  {row['method']:14s}  {row['n']:>4d}  {row['mean']:7.4f}  {row['std']:7.4f}  {ci:>16s}")


def print_snapshots(df: pd.DataFrame, snapshots: list[int]) -> None:
    max_evals = max(snapshots)
    print(f"\nBest-so-far P-Loss at eval counts: {snapshots}")
    header = f"  {'synth':18s}  {'method':14s}  " + "  ".join(f"@{s:>3d}" for s in snapshots)
    print(header)
    for synth in SYNTH_ORDER:
        for method in METHOD_ORDER:
            curve = mean_best_so_far(df, synth, method, max_evals)
            if np.all(np.isnan(curve)):
                continue
            vals = "  ".join(
                f"{curve[s-1]:6.4f}" if s <= len(curve) else "  --  "
                for s in snapshots
            )
            print(f"  {synth:18s}  {method:14s}  {vals}")


def print_npsk(df: pd.DataFrame) -> None:
    print("\nNPSK win-rate matrices (row beats column):")
    for synth in SYNTH_ORDER:
        mat = npsk_matrix(df, synth)
        if mat.isna().all().all():
            continue
        print(f"\n  {synth}")
        print("  " + mat.to_string().replace("\n", "\n  "))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_learning_curves(df: pd.DataFrame, save_dir: Path) -> None:
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)
    max_evals = int(df["eval_budget"].max())
    colors = {"GD": "black", "HillClimber": "tab:blue", "RandomSearch": "tab:orange",
              "CMA-ES": "tab:green", "BO": "tab:red"}

    for synth in SYNTH_ORDER:
        fig, ax = plt.subplots(figsize=(6, 4))
        for method in METHOD_ORDER:
            curve = mean_best_so_far(df, synth, method, max_evals)
            if np.all(np.isnan(curve)):
                continue
            ax.plot(np.arange(1, max_evals + 1), curve, label=method,
                    color=colors.get(method), linewidth=1.8)
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Best P-Loss (mean)")
        ax.set_title(synth)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = save_dir / f"curve_{synth}.pdf"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")


def plot_npsk_heatmap(df: pd.DataFrame, save_dir: Path) -> None:
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)
    synths = [s for s in SYNTH_ORDER if s in df["program"].unique()]
    fig, axes = plt.subplots(1, len(synths), figsize=(4 * len(synths), 3.5))
    if len(synths) == 1:
        axes = [axes]

    for ax, synth in zip(axes, synths):
        mat = npsk_matrix(df, synth)
        methods = mat.index.tolist()
        im = ax.imshow(mat.values.astype(float), vmin=0, vmax=1, cmap="RdYlGn")
        ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, fontsize=7)
        ax.set_title(synth, fontsize=8)
        for i in range(len(methods)):
            for j in range(len(methods)):
                v = mat.iloc[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("NPSK win rate (row beats column)", fontsize=9)
    fig.tight_layout()
    path = save_dir / "npsk_heatmap.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_boxplots(df: pd.DataFrame, save_dir: Path) -> None:
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)
    synths = [s for s in SYNTH_ORDER if s in df["program"].unique()]
    fig, axes = plt.subplots(1, len(synths), figsize=(4 * len(synths), 4), sharey=False)
    if len(synths) == 1:
        axes = [axes]

    for ax, synth in zip(axes, synths):
        data = []
        labels = []
        for method in METHOD_ORDER:
            vals = df[(df["program"] == synth) & (df["method"] == method)]["best_p_loss"].values
            if len(vals) > 0:
                data.append(vals)
                labels.append(method)
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(synth, fontsize=8)
        ax.set_ylabel("Best P-Loss")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = save_dir / "boxplots.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       type=Path, default=RESULTS_CSV)
    parser.add_argument("--snapshots", type=int,  nargs="+", default=SNAPSHOTS)
    parser.add_argument("--no-plots",  action="store_true")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Results file not found: {args.csv}")
        print("Run  python paper_experiments/run_paper.py  first.")
        sys.exit(1)

    df = load(args.csv)
    print(f"Loaded {len(df)} trials from {args.csv}")
    print(f"Synths:  {sorted(df['program'].unique())}")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Trials per cell: {df.groupby(['program','method']).size().to_dict()}")

    print_summary(df)
    print_snapshots(df, args.snapshots)
    print_npsk(df)

    if not args.no_plots:
        print("\nGenerating plots...")
        plot_learning_curves(df, FIGURES_DIR)
        plot_npsk_heatmap(df, FIGURES_DIR)
        plot_boxplots(df, FIGURES_DIR)
        print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
