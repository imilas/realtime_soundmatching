"""
Backend logic for paper_experiments/viz.py.

All computation and figure construction lives here.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_ORDER = ["GD", "HillClimber", "RandomSearch", "CMA-ES", "BO", "QL"]
COLORS = {
    "GD": "black",
    "HillClimber": "#1f77b4",
    "RandomSearch": "#ff7f0e",
    "CMA-ES": "#2ca02c",
    "BO": "#d62728",
    "QL": "#9467bd",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(results_dir: Path) -> pd.DataFrame:
    """Load results from CSV (fast) or rebuild from pkls and save CSV."""
    csv_path = results_dir / "results.csv"
    pkls = list(results_dir.glob("*.pkl"))
    if not pkls:
        return pd.DataFrame()
    total = sum(len(pickle.load(open(p, "rb"))["trials"]) for p in pkls)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if len(df) == total:
            return df
    rows: list[dict] = []
    for p in sorted(pkls):
        rows.extend(pickle.load(open(p, "rb"))["trials"])
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def keep_last_ql_pct(df: pd.DataFrame, pct: float) -> pd.DataFrame:
    """Keep only the last `pct`% of QL rows (Q-table warms up over trials)."""
    if df.empty:
        return df
    ql = df[df["method"] == "QL"]
    other = df[df["method"] != "QL"]
    if ql.empty:
        return df
    kept = []
    for _synth, rows in ql.groupby("program", sort=False):
        n = max(1, int(np.ceil(len(rows) * pct / 100.0)))
        kept.append(rows.tail(n))
    return pd.concat([other, *kept], ignore_index=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_boxplots(filtered: pd.DataFrame, method_order=METHOD_ORDER, colors=COLORS) -> plt.Figure:
    """Best P-Loss boxplots per synth × method. Returns a Figure."""
    synths = sorted(filtered["program"].unique())
    methods = [m for m in method_order if m in filtered["method"].unique()]
    fig, axes = plt.subplots(1, len(synths), figsize=(4 * len(synths), 4), squeeze=False)
    for ax, synth in zip(axes[0], synths):
        data, labels, cols = [], [], []
        for m in methods:
            vals = filtered[(filtered["program"] == synth) & (filtered["method"] == m)]["best_p_loss"].values
            if len(vals):
                data.append(vals); labels.append(m); cols.append(colors.get(m, "gray"))
        bp = ax.boxplot(data, patch_artist=True, showfliers=True)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(synth, fontsize=9)
        ax.set_ylabel("Best P-Loss")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Best P-Loss distribution per method", fontsize=11)
    fig.tight_layout()
    return fig


def plot_ql_learning(selected: pd.DataFrame, colors=COLORS) -> plt.Figure | None:
    """QL best P-Loss across trials (moving mean). Returns a Figure or None if no QL data."""
    ql = selected[selected["method"] == "QL"].copy()
    if ql.empty:
        return None
    synths = sorted(ql["program"].unique())
    fig, axes = plt.subplots(1, len(synths), figsize=(4 * len(synths), 3.5), squeeze=False)
    for ax, synth in zip(axes[0], synths):
        vals = ql[ql["program"] == synth]["best_p_loss"].values
        w = max(1, min(250, len(vals) // 20))
        smooth = np.convolve(vals, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(w, len(vals) + 1), smooth, color=colors["QL"], lw=2,
                label=f"moving mean (w={w})")
        ax.set_xlabel("Trial #"); ax.set_ylabel("Best P-Loss")
        ax.set_title(f"QL — {synth}", fontsize=9)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.suptitle("QL: P-Loss across trials (Q-table persists)", fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def compute_summary_table(filtered: pd.DataFrame, method_order=METHOD_ORDER) -> pd.DataFrame:
    """Mean / std / min best P-Loss per (synth, method)."""
    rows = []
    for synth in sorted(filtered["program"].unique()):
        for m in [x for x in method_order if x in filtered["method"].unique()]:
            vals = filtered[(filtered["program"] == synth) & (filtered["method"] == m)]["best_p_loss"].values
            if len(vals):
                rows.append({
                    "synth": synth, "method": m, "n": len(vals),
                    "mean": round(float(np.mean(vals)), 4),
                    "std":  round(float(np.std(vals)), 4),
                    "min":  round(float(np.min(vals)), 4),
                })
    return pd.DataFrame(rows)
