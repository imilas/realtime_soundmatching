"""
Plotting and utility helpers for fr_v2.py.

Edit this file freely — the notebook reloads it on each plot cell run
via importlib.reload, so data never reloads when you change a plot.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

def build_method_colors(all_methods: list[str]) -> dict[str, str]:
    BASE = {"CMA-ES": "#3cb44b", "LES": "#4363d8", "RandomSearch": "#f58231"}
    gd_sorted = sorted(
        [m for m in all_methods if m == "GD" or m.startswith("GD_lr")],
        key=lambda m: float(m[5:]) if m.startswith("GD_lr") else 0.045,
    )
    reds = plt.cm.Reds(np.linspace(0.35, 0.9, max(len(gd_sorted), 1)))
    gd_colors = {m: "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
                 for m, (r, g, b, _) in zip(gd_sorted, reds)}
    return {**BASE, **gd_colors}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def fig_to_html(fig, dpi: int = 72, min_width: int = 1000) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return (
        f'<div style="overflow-x:auto">'
        f'<img src="data:image/png;base64,{b64}" style="min-width:{min_width}px">'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def sample_efficiency_plot(
    trial_cache: dict,
    synths: list[str],
    losses: list[str],
    methods: list[str],
    synth_labels: dict[str, str],
    method_colors: dict[str, str],
) -> plt.Figure:
    """
    Grid: rows=losses, cols=synths.
    Each subplot shows median best-so-far P-loss (fmin.accumulate) per method.
    """
    n_rows, n_cols = len(losses), len(synths)
    plt.rcParams.update({"font.size": 30})
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(8.0 * n_cols, 6.4 * n_rows),
        squeeze=False,
    )

    for ri, loss in enumerate(losses):
        for ci, synth in enumerate(synths):
            ax = axes[ri][ci]
            any_plotted = False

            for method in methods:
                trials = trial_cache.get((synth, loss, method), [])
                if not trials:
                    continue
                B = max((len(t.get("history_p_loss", [])) for t in trials), default=0)
                if B == 0:
                    continue

                curves = []
                for t in trials:
                    c = np.fmin.accumulate(
                        np.asarray(t.get("history_p_loss", []), dtype=float)
                    )
                    if len(c) < B:
                        c = np.pad(c, (0, B - len(c)),
                                   constant_values=c[-1] if len(c) else np.nan)
                    curves.append(c)

                med = np.nanmedian(np.vstack(curves), axis=0)
                ax.plot(
                    np.arange(1, B + 1), med,
                    color=method_colors.get(method, "#888888"),
                    lw=1.6,
                    label=method.replace("GD_lr", "lr="),
                )
                any_plotted = True

            if ri == n_rows - 1:
                ax.set_xlabel("evaluations", fontsize=15)
            if ci == 0:
                ax.set_ylabel(f"{loss}\nbest-so-far P-loss (log)", fontsize=30)
            if ri == 0:
                ax.set_title(synth_labels.get(synth, synth), fontsize=30)

            ax.grid(True, alpha=0.3, which="both")
            if any_plotted:
                ax.set_yscale("log")
            if ri == 0 and ci == 0 and any_plotted:
                ax.legend(fontsize=15)

    fig.suptitle(
        "Sample efficiency: median best-so-far P-loss, rows=loss, cols=synth",
        fontsize=30,
    )
    fig.tight_layout()
    return fig


def returned_ploss_plot(
    trial_cache: dict,
    synths: list[str],
    losses: list[str],
    methods: list[str],
    synth_labels: dict[str, str],
    method_colors: dict[str, str],
) -> plt.Figure:
    """
    Grid: rows=losses, cols=synths.
    Each subplot shows median instantaneous P-loss (raw history_p_loss) per method.
    """
    n_rows, n_cols = len(losses), len(synths)
    plt.rcParams.update({"font.size": 30})
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(8.0 * n_cols, 6.4 * n_rows),
        squeeze=False,
    )

    for ri, loss in enumerate(losses):
        for ci, synth in enumerate(synths):
            ax = axes[ri][ci]
            any_plotted = False

            for method in methods:
                trials = trial_cache.get((synth, loss, method), [])
                if not trials:
                    continue
                B = max((len(t.get("history_p_loss", [])) for t in trials), default=0)
                if B == 0:
                    continue

                curves = []
                for t in trials:
                    c = np.asarray(t.get("history_p_loss", []), dtype=float)
                    if len(c) < B:
                        c = np.pad(c, (0, B - len(c)), constant_values=np.nan)
                    curves.append(c)

                med = np.nanmedian(np.vstack(curves), axis=0)
                ax.plot(
                    np.arange(1, B + 1), med,
                    color=method_colors.get(method, "#888888"),
                    lw=1.6,
                    label=method.replace("GD_lr", "lr="),
                )
                any_plotted = True

            if ri == n_rows - 1:
                ax.set_xlabel("evaluations", fontsize=15)
            if ci == 0:
                ax.set_ylabel(f"{loss}\nreturned P-loss", fontsize=30)
            if ri == 0:
                ax.set_title(synth_labels.get(synth, synth), fontsize=30)

            ax.grid(True, alpha=0.3, which="both")
            if ri == 0 and ci == 0 and any_plotted:
                ax.legend(fontsize=15)

    fig.suptitle(
        "Sanity check: median returned (instantaneous) P-loss, rows=loss, cols=synth",
        fontsize=30,
    )
    fig.tight_layout()
    return fig


def audio_loss_plot(
    trial_cache: dict,
    synths: list[str],
    losses: list[str],
    methods: list[str],
    synth_labels: dict[str, str],
    method_colors: dict[str, str],
) -> plt.Figure:
    """
    Grid: rows=losses, cols=synths.
    Each subplot shows median audio loss curves (history_audio_loss) per method.
    """
    n_rows, n_cols = len(losses), len(synths)
    plt.rcParams.update({"font.size": 30})
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(8.0 * n_cols, 6.4 * n_rows),
        squeeze=False,
    )

    for ri, loss in enumerate(losses):
        for ci, synth in enumerate(synths):
            ax = axes[ri][ci]
            any_plotted = False

            for method in methods:
                trials = trial_cache.get((synth, loss, method), [])
                if not trials:
                    continue
                B = max((len(t.get("history_audio_loss", [])) for t in trials), default=0)
                if B == 0:
                    continue

                curves = []
                for t in trials:
                    c = np.asarray(t.get("history_audio_loss", []), dtype=float)
                    if len(c) < B:
                        c = np.pad(c, (0, B - len(c)), constant_values=np.nan)
                    curves.append(c)

                med = np.nanmedian(np.vstack(curves), axis=0)
                ax.plot(
                    np.arange(1, B + 1), med,
                    color=method_colors.get(method, "#888888"),
                    lw=1.6,
                    label=method.replace("GD_lr", "lr="),
                )
                any_plotted = True

            if ri == n_rows - 1:
                ax.set_xlabel("evaluations", fontsize=15)
            if ci == 0:
                ax.set_ylabel(f"{loss}\naudio loss", fontsize=30)
            if ri == 0:
                ax.set_title(synth_labels.get(synth, synth), fontsize=30)

            ax.grid(True, alpha=0.3, which="both")
            if ri == 0 and ci == 0 and any_plotted:
                ax.legend(fontsize=15)

    fig.suptitle(
        "Median audio-loss curves, rows=loss, cols=synth",
        fontsize=30,
    )
    fig.tight_layout()
    return fig


def audio_loss_best_so_far_plot(
    trial_cache: dict,
    synths: list[str],
    losses: list[str],
    methods: list[str],
    synth_labels: dict[str, str],
    method_colors: dict[str, str],
) -> plt.Figure:
    """
    Grid: rows=losses, cols=synths.
    Each subplot shows median best-so-far audio loss (fmin.accumulate) per method.
    """
    n_rows, n_cols = len(losses), len(synths)
    plt.rcParams.update({"font.size": 30})
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(8.0 * n_cols, 6.4 * n_rows),
        squeeze=False,
    )

    for ri, loss in enumerate(losses):
        for ci, synth in enumerate(synths):
            ax = axes[ri][ci]
            any_plotted = False

            for method in methods:
                trials = trial_cache.get((synth, loss, method), [])
                if not trials:
                    continue
                B = max((len(t.get("history_audio_loss", [])) for t in trials), default=0)
                if B == 0:
                    continue

                curves = []
                for t in trials:
                    c = np.fmin.accumulate(
                        np.asarray(t.get("history_audio_loss", []), dtype=float)
                    )
                    if len(c) < B:
                        c = np.pad(c, (0, B - len(c)),
                                   constant_values=c[-1] if len(c) else np.nan)
                    curves.append(c)

                med = np.nanmedian(np.vstack(curves), axis=0)
                ax.plot(
                    np.arange(1, B + 1), med,
                    color=method_colors.get(method, "#888888"),
                    lw=1.6,
                    label=method.replace("GD_lr", "lr="),
                )
                any_plotted = True

            if ri == n_rows - 1:
                ax.set_xlabel("evaluations", fontsize=15)
            if ci == 0:
                ax.set_ylabel(f"{loss}\nbest-so-far audio loss", fontsize=30)
            if ri == 0:
                ax.set_title(synth_labels.get(synth, synth), fontsize=30)

            ax.grid(True, alpha=0.3, which="both")
            if any_plotted:
                ax.set_yscale("log")
            if ri == 0 and ci == 0 and any_plotted:
                ax.legend(fontsize=15)

    fig.suptitle(
        "Median best-so-far audio-loss curves, rows=loss, cols=synth",
        fontsize=30,
    )
    fig.tight_layout()
    return fig
