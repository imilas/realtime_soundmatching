"""
Backend logic for trajectory_deep_dive.py.

All computation lives here; the notebook just wires up UI and calls these.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent.params import FaustParams
from experiments.multidim_runner import _bounds_from_params, _render_audio
from paper_experiments.config import AUDIO_DURATION_S, SAMPLE_RATE, SYNTH_LOSS
from synths.build import prepare
from utils.loss_functions import ALL_LOSSES

METHOD_ORDER = ["GD", "RandomSearch", "CMA-ES", "BO"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(results_dir: Path) -> dict:
    """Return {synth: {method: pkl_data}} for every *_*.pkl in results_dir."""
    out: dict = {}
    for path in sorted(results_dir.glob("*.pkl")):
        stem = path.stem
        if "_" not in stem:
            continue
        synth, method = stem.rsplit("_", 1)
        with open(path, "rb") as f:
            out.setdefault(synth, {})[method] = pickle.load(f)
    return out


# ---------------------------------------------------------------------------
# Trial context
# ---------------------------------------------------------------------------

def build_trial_context(all_results: dict, synth: str, method: str, trial_index: int) -> dict:
    """
    Load everything needed to visualise one trial.

    Returns a dict with keys:
        trial, trials, trial_index, synth, method,
        build, params, bounds, names, normalize
    where `normalize(param_dict)` converts raw params to [0,1].
    """
    data = all_results[synth][method]
    trials = data.get("trials", [])
    idx = int(np.clip(trial_index, 0, max(0, len(trials) - 1)))
    trial = trials[idx] if trials else {}

    build = prepare(synth)
    params = FaustParams(str(build.json_path))
    bounds = _bounds_from_params(params)
    names = params.names()

    def normalize(param_dict: dict) -> np.ndarray:
        real = np.array([float(param_dict[n]) for n in names], dtype=float)
        return bounds.normalize(real)

    return dict(
        trial=trial, trials=trials, trial_index=idx,
        synth=synth, method=method,
        build=build, params=params, bounds=bounds, names=names,
        normalize=normalize,
    )


def extract_trajectory(ctx: dict) -> dict:
    """
    Pull history arrays and normalised key points out of a trial context.

    Returns a dict with keys:
        trajectory, true_norm, init_norm, best_norm,
        history_p_loss, history_audio_loss
    """
    trial = ctx["trial"]
    normalize = ctx["normalize"]
    hist = trial.get("history_params", []) if trial else []
    return dict(
        trajectory=(
            np.array([normalize(p) for p in hist], dtype=float)
            if hist else np.empty((0, 0), dtype=float)
        ),
        true_norm=normalize(trial["true_params"]) if trial else np.array([]),
        init_norm=normalize(trial["init_params"]) if trial else np.array([]),
        best_norm=normalize(trial["best_params"]) if trial else np.array([]),
        history_p_loss=trial.get("history_p_loss", []) if trial else [],
        history_audio_loss=trial.get("history_audio_loss", []) if trial else [],
    )


# ---------------------------------------------------------------------------
# Surface computation
# ---------------------------------------------------------------------------

def compute_surface(mode: str, grid_n: int, ctx: dict, traj: dict) -> tuple:
    """
    Return (xx, yy, surface, label).

    mode: "P-Loss surface" or "Audio-loss surface"
    """
    xs = np.linspace(0.0, 1.0, grid_n)
    xx, yy = np.meshgrid(xs, xs)
    true_norm = traj["true_norm"]

    if not ctx["trial"] or true_norm.size < 2:
        return xx, yy, np.zeros_like(xx), "n/a"

    if mode == "P-Loss surface":
        surface = np.sqrt((xx - true_norm[0]) ** 2 + (yy - true_norm[1]) ** 2)
        return xx, yy, surface, "P-Loss distance to true params"

    # Audio-loss surface — expensive
    synth = ctx["synth"]
    loss_fn = ALL_LOSSES[SYNTH_LOSS[synth]]
    n_samples = int(AUDIO_DURATION_S * SAMPLE_RATE)
    target = _render_audio(str(ctx["build"].dsp_path), ctx["trial"]["true_params"], n_samples, SAMPLE_RATE)
    surface = np.zeros_like(xx)
    for row in range(grid_n):
        for col in range(grid_n):
            norm = np.array([xx[row, col], yy[row, col]], dtype=float)
            real = ctx["bounds"].denormalize(norm)
            audio = _render_audio(str(ctx["build"].dsp_path), ctx["params"].vector_to_dict(real), n_samples, SAMPLE_RATE)
            m = min(len(audio), len(target))
            surface[row, col] = float(loss_fn(target[:m], audio[:m], sample_rate=SAMPLE_RATE))
    return xx, yy, surface, f"Audio loss: {SYNTH_LOSS[synth]}"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _draw_surface(ax, xx, yy, surface, surface_label):
    c = ax.contourf(xx, yy, surface, levels=24, cmap="viridis")
    plt.gcf().colorbar(c, ax=ax, label=surface_label)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(color="white", alpha=0.25)


def plot_full_trajectory(traj: dict, ctx: dict, xx, yy, surface, surface_label) -> plt.Figure:
    """Full trajectory overlaid on loss surface. Returns a Figure."""
    trajectory = traj["trajectory"]
    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_surface(ax, xx, yy, surface, surface_label)
    names = ctx["names"]

    ax.plot(trajectory[:, 0], trajectory[:, 1], color="white", lw=1.3, alpha=0.9, label="trajectory")
    ax.scatter(
        trajectory[:, 0], trajectory[:, 1],
        c=np.arange(len(trajectory)), cmap="plasma", s=18,
        edgecolor="black", linewidth=0.25, label="evals",
    )
    ax.scatter(*traj["init_norm"][:2], marker="s", s=90, color="cyan",  edgecolor="black", label="init")
    ax.scatter(*traj["true_norm"][:2], marker="*", s=220, color="red",  edgecolor="black", label="true")
    ax.scatter(*traj["best_norm"][:2], marker="X", s=110, color="lime", edgecolor="black", label="best")
    ax.set_xlabel(f"{names[0]} (norm)"); ax.set_ylabel(f"{names[1]} (norm)")
    ax.set_title(f"{ctx['method']} — full trajectory")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_step_trajectory(traj: dict, ctx: dict, xx, yy, surface, surface_label, step_k: int) -> plt.Figure:
    """First step_k evals overlaid on loss surface. Returns a Figure."""
    trajectory = traj["trajectory"]
    sub = trajectory[:step_k]
    cur = sub[-1]
    p_loss = traj["history_p_loss"]
    p = p_loss[step_k - 1] if step_k - 1 < len(p_loss) else float("nan")
    names = ctx["names"]

    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_surface(ax, xx, yy, surface, surface_label)
    ax.plot(sub[:, 0], sub[:, 1], color="white", lw=1.3, alpha=0.9)
    ax.scatter(
        sub[:, 0], sub[:, 1],
        c=np.arange(step_k), cmap="plasma", s=18,
        edgecolor="black", linewidth=0.25,
    )
    ax.scatter(*traj["init_norm"][:2], marker="s", s=90, color="cyan",  edgecolor="black", label="init")
    ax.scatter(*traj["true_norm"][:2], marker="*", s=220, color="red",  edgecolor="black", label="true")
    ax.scatter(*traj["best_norm"][:2], marker="X", s=110, color="lime", edgecolor="black", label="best")
    ax.scatter(cur[0], cur[1], marker="o", s=140, facecolor="none",
               edgecolor="yellow", lw=2.2, label=f"eval {step_k - 1}")
    ax.set_xlabel(f"{names[0]} (norm)"); ax.set_ylabel(f"{names[1]} (norm)")
    ax.set_title(f"{ctx['method']} — eval {step_k - 1}/{len(trajectory) - 1}  |  P-Loss={p:.4f}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_loss_curves(traj: dict, step_k: int) -> plt.Figure:
    """P-Loss and audio-loss over all evals, with step_k marker. Returns a Figure."""
    hp = traj["history_p_loss"]
    ha = traj["history_audio_loss"]
    x = np.arange(len(hp))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    for ax, vals, color, title, ylabel in [
        (axes[0], hp, "#1f77b4", "P-Loss per evaluation", "P-Loss"),
        (axes[1], ha, "#d62728", "Audio loss per evaluation", "Audio loss"),
    ]:
        ax.plot(x, vals, color=color, alpha=0.35)
        ax.plot(x[:step_k], vals[:step_k], color=color, lw=1.8)
        ax.axvline(step_k - 1, color="gray", ls="--", alpha=0.6)
        ax.set_title(title); ax.set_xlabel("Evaluation"); ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def make_eval_table(traj: dict) -> pd.DataFrame:
    """DataFrame of per-eval normalised coords and losses."""
    trajectory = traj["trajectory"]
    hp = traj["history_p_loss"]
    ha = traj["history_audio_loss"]
    return pd.DataFrame([
        {
            "eval": i,
            "x_norm": round(float(pt[0]), 4),
            "y_norm": round(float(pt[1]), 4) if pt.shape[0] > 1 else None,
            "p_loss": round(float(hp[i]), 5),
            "audio_loss": round(float(ha[i]), 5),
        }
        for i, pt in enumerate(trajectory)
    ])


def trial_summary_md(ctx: dict, traj: dict) -> str:
    """Markdown string summarising the selected trial."""
    trial = ctx["trial"]
    hp = traj["history_p_loss"]
    ql_size = trial.get("ql_q_table_size")
    ql_eps = trial.get("ql_epsilon_end")
    ql_line = (
        f"\n\nQ-table size: `{ql_size}` &nbsp;|&nbsp; Epsilon: `{ql_eps}`"
        if ql_size is not None else ""
    )
    return (
        f"**{ctx['method']} — Trial {ctx['trial_index']} / {max(0, len(ctx['trials']) - 1)}**\n\n"
        f"Loss: `{trial.get('loss_name', '?')}` &nbsp;|&nbsp; "
        f"Parameters: `{', '.join(ctx['names'])}`\n\n"
        f"Evaluations: `{len(hp)}` &nbsp;|&nbsp; "
        f"Best P-Loss: `{trial.get('best_p_loss', float('nan')):.4f}` &nbsp;|&nbsp; "
        f"Duration: `{trial.get('duration_s', float('nan')):.1f}s`{ql_line}"
    )
