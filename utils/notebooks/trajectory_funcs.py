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

METHOD_ORDER = ["GD", "RandomSearch", "CMA-ES"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(results_dir: Path) -> dict:
    """Return {synth: {method: {loss_name: pkl_data}}} for every *_*.pkl in results_dir.

    Handles both old naming ({synth}_{method}.pkl) and new naming
    ({synth}_{loss}_{method}.pkl). All loss variants for a (synth, method)
    pair are kept so the notebook can offer a loss picker.
    """
    from synths.program import list_programs
    known_synths = sorted(list_programs(), key=len, reverse=True)
    out: dict = {}
    for path in sorted(results_dir.glob("*.pkl")):
        stem = path.stem
        synth = next((s for s in known_synths if stem.startswith(s + "_")), None)
        if synth is None:
            continue
        remainder = stem[len(synth) + 1:]  # "loss_method" or "method"
        method = remainder.rsplit("_", 1)[-1]
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Infer loss name from the data itself (stored in each trial)
        trials = data.get("trials", [])
        loss_name = trials[0].get("loss_name", "unknown") if trials else remainder[: -(len(method) + 1)] or "unknown"
        out.setdefault(synth, {}).setdefault(method, {})[loss_name] = data
    return out


# ---------------------------------------------------------------------------
# Trial context
# ---------------------------------------------------------------------------

def build_trial_context(all_results: dict, synth: str, method: str, trial_index: int, loss: str | None = None) -> dict:
    """
    Load everything needed to visualise one trial.

    Returns a dict with keys:
        trial, trials, trial_index, synth, method, loss,
        build, params, bounds, names, normalize
    where `normalize(param_dict)` converts raw params to [0,1].
    """
    loss_map = all_results[synth][method]  # {loss_name: pkl_data}
    if loss is None or loss not in loss_map:
        loss = next(iter(loss_map))
    data = loss_map[loss]
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
        synth=synth, method=method, loss=loss,
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
    loss_key = ctx.get("loss", SYNTH_LOSS[synth])
    loss_fn = ALL_LOSSES.get(loss_key) or ALL_LOSSES[SYNTH_LOSS[synth]]
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

def _draw_surface(ax, xx, yy, surface, surface_label, norm_mode="Linear"):
    import matplotlib.colors as mcolors
    s = surface.copy()
    if norm_mode == "Log":
        vmin = max(s[s > 0].min(), 1e-10) if (s > 0).any() else 1e-10
        norm = mcolors.LogNorm(vmin=vmin, vmax=s.max())
    elif norm_mode == "Clip 95th pct":
        norm = mcolors.Normalize(vmin=s.min(), vmax=np.percentile(s, 95))
    else:
        norm = mcolors.Normalize(vmin=s.min(), vmax=s.max())
    c = ax.contourf(xx, yy, s, levels=24, cmap="viridis", norm=norm)
    plt.gcf().colorbar(c, ax=ax, label=surface_label)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(color="white", alpha=0.25)


def skip_indices(n_total: int, skip_pct: float, keep_first: int = 10) -> np.ndarray:
    """Indices to display: first `keep_first` always, then downsample the rest.

    skip_pct=20 keeps 80% of evals after the first `keep_first`.
    """
    head = np.arange(min(keep_first, n_total))
    if n_total <= keep_first:
        return head
    rest = np.arange(keep_first, n_total)
    n_keep = max(1, int(len(rest) * (1.0 - skip_pct / 100.0)))
    kept = rest[np.round(np.linspace(0, len(rest) - 1, n_keep)).astype(int)]
    return np.concatenate([head, kept])


_LABEL_Y_OFFSET = {"init": 1.04, "true": 1.09, "best": 1.14}

def _label_marker(ax, x, y, text, color):
    ty = _LABEL_Y_OFFSET.get(text, 1.04)
    ax.annotate(text, xy=(x, y), xytext=(x, ty),
                ha="center", va="bottom", fontsize=8, color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=color, alpha=0.5, lw=0.8),
                zorder=6, annotation_clip=False)


def plot_full_trajectory(traj: dict, ctx: dict, xx, yy, surface, surface_label,
                         indices: "np.ndarray | None" = None,
                         norm_mode: str = "Linear") -> plt.Figure:
    trajectory = traj["trajectory"]
    disp = trajectory[indices] if indices is not None else trajectory
    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_surface(ax, xx, yy, surface, surface_label, norm_mode=norm_mode)
    names = ctx["names"]

    ax.plot(disp[:, 0], disp[:, 1], color="white", lw=1.3, alpha=0.9, label="trajectory", zorder=2)
    ax.scatter(disp[:, 0], disp[:, 1], c=np.arange(len(disp)), cmap="plasma", s=18,
               edgecolor="black", linewidth=0.25, label="evals", zorder=3)

    for pt, marker, color, size, label in [
        (traj["init_norm"][:2], "s", "cyan",  90,  "init"),
        (traj["true_norm"][:2], "*", "red",   220, "true"),
        (traj["best_norm"][:2], "X", "lime",  110, "best"),
    ]:
        ax.scatter(*pt, marker=marker, s=size, color=color, edgecolor="black", zorder=5, label=label)
        _label_marker(ax, pt[0], pt[1], label, color)

    ax.set_xlabel(f"{names[0]} (norm)"); ax.set_ylabel(f"{names[1]} (norm)")
    ax.set_title(f"{ctx['method']} — full trajectory")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_step_trajectory(traj: dict, ctx: dict, xx, yy, surface, surface_label,
                         step_k: int, norm_mode: str = "Linear") -> plt.Figure:
    trajectory = traj["trajectory"]
    sub = trajectory[:step_k]
    cur = sub[-1]
    p_loss = traj["history_p_loss"]
    p = p_loss[step_k - 1] if step_k - 1 < len(p_loss) else float("nan")
    names = ctx["names"]

    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_surface(ax, xx, yy, surface, surface_label, norm_mode=norm_mode)
    ax.plot(sub[:, 0], sub[:, 1], color="white", lw=1.3, alpha=0.9, zorder=2)
    ax.scatter(sub[:, 0], sub[:, 1], c=np.arange(step_k), cmap="plasma", s=18,
               edgecolor="black", linewidth=0.25, zorder=3)

    for pt, marker, color, size, label in [
        (traj["init_norm"][:2], "s", "cyan",  90,  "init"),
        (traj["true_norm"][:2], "*", "red",   220, "true"),
        (traj["best_norm"][:2], "X", "lime",  110, "best"),
    ]:
        ax.scatter(*pt, marker=marker, s=size, color=color, edgecolor="black", zorder=5, label=label)
        _label_marker(ax, pt[0], pt[1], label, color)

    ax.scatter(cur[0], cur[1], marker="o", s=140, facecolor="none",
               edgecolor="yellow", lw=2.2, label=f"eval {step_k - 1}", zorder=5)
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


def render_trial_audios(ctx: dict) -> "list[tuple[str, np.ndarray, dict]]":
    """Render true / returned / min-P-loss / min-audio-loss audio for the selected trial.

    Returns a list of (label, audio_array, param_dict) in display order.
    Returns [] if no trial is selected.
    """
    trial = ctx["trial"]
    if not trial:
        return []
    n_samples = int(AUDIO_DURATION_S * SAMPLE_RATE)
    dsp = str(ctx["build"].dsp_path)

    def _entry(label, params):
        return (label, _render_audio(dsp, params, n_samples, SAMPLE_RATE), params)

    out = [_entry("True", trial["true_params"])]

    hist = trial.get("history_params", [])
    al = trial.get("history_audio_loss", [])
    if hist:
        if ctx["method"] == "GD":
            out.append(_entry("Returned (final GD step)", hist[-1]))
        else:
            ret_idx = int(np.argmin(al)) if al else len(hist) - 1
            out.append(_entry("Returned (min-audio step)", hist[ret_idx]))

        out.append(_entry("Min P-loss", trial["best_params"]))

        al_idx = int(np.argmin(al)) if al else len(hist) - 1
        out.append(_entry("Min audio-loss", hist[al_idx]))

    return out


def audio_to_wav_bytes(audio: np.ndarray):
    """Convert a float audio array to a WAV BytesIO (16-bit PCM)."""
    import io
    from scipy.io.wavfile import write as wav_write
    buf = io.BytesIO()
    arr = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    wav_write(buf, SAMPLE_RATE, arr)
    buf.seek(0)
    return buf


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


def fill_dsp_params(dsp_source: str, params: dict) -> str:
    """Replace hslider default values with the given param dict values."""
    import re
    def _sub(m):
        name = m.group(1)
        if name in params:
            return f'hslider("{name}",{params[name]},{m.group(3)},{m.group(4)},{m.group(5)})'
        return m.group(0)
    return re.sub(r'hslider\("([^"]+)",([^,]+),([^,]+),([^,]+),([^)]+)\)', _sub, dsp_source)


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
