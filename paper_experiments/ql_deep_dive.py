import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import pickle
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from agent.params import FaustParams
    from experiments.multidim_runner import _bounds_from_params, _render_audio
    from paper_experiments.config import AUDIO_DURATION_S, SAMPLE_RATE, SYNTH_LOSS
    from synths.build import prepare
    from utils.loss_functions import ALL_LOSSES

    RESULTS_DIR = Path("paper_experiments/results")
    return (
        ALL_LOSSES,
        AUDIO_DURATION_S,
        FaustParams,
        RESULTS_DIR,
        SAMPLE_RATE,
        SYNTH_LOSS,
        mo,
        np,
        pd,
        pickle,
        plt,
        prepare,
    )


@app.cell
def _(RESULTS_DIR, pickle):
    def load_ql_results(results_dir):
        out = {}
        for path in sorted(results_dir.glob("*_QL.pkl")):
            synth = path.name.removesuffix("_QL.pkl")
            with open(path, "rb") as f:
                data = pickle.load(f)
            out[synth] = data
        return out

    ql_results = load_ql_results(RESULTS_DIR)
    synths = sorted(ql_results)
    return ql_results, synths


@app.cell
def _(mo, ql_results, synths):
    if not synths:
        mo.md("No QL pickle files found in `paper_experiments/results`.")
    else:
        synth_picker = mo.ui.dropdown(
            options=synths,
            value=synths[0],
            label="Synth",
        )
        max_trial = max(
            max(0, len(ql_results[s].get("trials", [])) - 1)
            for s in synths
        )
        trial_picker = mo.ui.number(
            start=0,
            stop=max_trial,
            step=1,
            value=0,
            label="Trial index",
        )
        surface_picker = mo.ui.dropdown(
            options=["P-Loss surface", "Audio-loss surface"],
            value="P-Loss surface",
            label="Surface",
        )
        grid_slider = mo.ui.slider(
            start=15,
            stop=61,
            step=2,
            value=31,
            label="Grid resolution",
        )
        mo.hstack([synth_picker, trial_picker, surface_picker, grid_slider])
    return grid_slider, surface_picker, synth_picker, trial_picker


@app.cell
def _(FaustParams, np, prepare, ql_results, synth_picker, trial_picker):
    synth = synth_picker.value
    data = ql_results[synth]
    trials = data.get("trials", [])
    trial_index = int(np.clip(trial_picker.value, 0, max(0, len(trials) - 1)))
    trial = trials[trial_index] if trials else {}

    build = prepare(synth)
    params = FaustParams(str(build.json_path))
    bounds = _bounds_from_params(params)
    names = params.names()

    def normalize_params(param_dict):
        real = np.array([float(param_dict[name]) for name in names], dtype=float)
        return bounds.normalize(real)

    return (
        bounds,
        build,
        names,
        normalize_params,
        params,
        synth,
        trial,
        trial_index,
        trials,
    )


@app.cell
def _(mo, names, trial, trial_index, trials):
    if not trial:
        mo.md("No selected trial.")
    else:
        history_params = trial.get("history_params", [])
        ql_size = trial.get("ql_q_table_size")
        ql_eps = trial.get("ql_epsilon_end")
        mo.md(
            f"""
            **Trial {trial_index} / {max(0, len(trials) - 1)}**

            Parameters: `{", ".join(names)}`

            Evaluations in trajectory: `{len(history_params)}`

            Best P-Loss: `{trial.get("best_p_loss", float("nan")):.4f}`

            Q-table size at end of trial: `{ql_size}`

            Epsilon at end of trial: `{ql_eps}`
            """
        )
    return


@app.cell
def _(normalize_params, np, trial):
    history_params = trial.get("history_params", []) if trial else []
    history_audio_loss = trial.get("history_audio_loss", []) if trial else []
    history_p_loss = trial.get("history_p_loss", []) if trial else []

    trajectory = (
        np.array([normalize_params(p) for p in history_params], dtype=float)
        if history_params
        else np.empty((0, 0), dtype=float)
    )
    true_norm = normalize_params(trial["true_params"]) if trial else np.array([])
    init_norm = normalize_params(trial["init_params"]) if trial else np.array([])
    best_norm = normalize_params(trial["best_params"]) if trial else np.array([])
    return (
        best_norm,
        history_audio_loss,
        history_p_loss,
        init_norm,
        trajectory,
        true_norm,
    )


@app.cell
def _(
    ALL_LOSSES,
    AUDIO_DURATION_S,
    SAMPLE_RATE,
    SYNTH_LOSS,
    bounds,
    build,
    grid_slider,
    np,
    params,
    surface_picker,
    synth,
    trial,
    true_norm,
):
    grid_n = int(grid_slider.value)
    xs = np.linspace(0.0, 1.0, grid_n)
    ys = np.linspace(0.0, 1.0, grid_n)
    xx, yy = np.meshgrid(xs, ys)

    if surface_picker.value == "P-Loss surface":
        surface = np.sqrt((xx - true_norm[0]) ** 2 + (yy - true_norm[1]) ** 2)
        surface_label = "P-Loss to true params"
    else:
        loss_name = SYNTH_LOSS[synth]
        loss_fn = ALL_LOSSES[loss_name]
        n_samples = int(AUDIO_DURATION_S * SAMPLE_RATE)
        target_audio = _render_audio(
            str(build.dsp_path),
            trial["true_params"],
            n_samples,
            SAMPLE_RATE,
        )
        surface = np.zeros_like(xx)
        for row in range(grid_n):
            for col in range(grid_n):
                norm = np.array([xx[row, col], yy[row, col]], dtype=float)
                real = bounds.denormalize(norm)
                candidate_params = params.vector_to_dict(real)
                audio = _render_audio(
                    str(build.dsp_path),
                    candidate_params,
                    n_samples,
                    SAMPLE_RATE,
                )
                m = min(len(audio), len(target_audio))
                surface[row, col] = float(
                    loss_fn(target_audio[:m], audio[:m], sample_rate=SAMPLE_RATE)
                )
        surface_label = f"Audio loss: {loss_name}"
    return surface, surface_label, xx, yy


@app.cell
def _(
    best_norm,
    init_norm,
    mo,
    names,
    np,
    plt,
    surface,
    surface_label,
    trajectory,
    true_norm,
    xx,
    yy,
):
    if trajectory.size == 0:
        mo.md(
            "This trial has no `history_params`. Rerun QL with the updated runner to save trajectories."
        )
    elif trajectory.shape[1] != 2:
        mo.md("This visualization currently supports 2-parameter synths only.")
    else:
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        contour = ax.contourf(xx, yy, surface, levels=24, cmap="viridis")
        fig.colorbar(contour, ax=ax, label=surface_label)

        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color="white",
            linewidth=1.3,
            alpha=0.9,
            label="trajectory",
        )
        ax.scatter(
            trajectory[:, 0],
            trajectory[:, 1],
            c=np.arange(len(trajectory)),
            cmap="plasma",
            s=18,
            edgecolor="black",
            linewidth=0.25,
            label="evals",
        )
        ax.scatter(*init_norm[:2], marker="s", s=90, color="cyan", edgecolor="black", label="init")
        ax.scatter(*true_norm[:2], marker="*", s=220, color="red", edgecolor="black", label="true")
        ax.scatter(*best_norm[:2], marker="X", s=110, color="lime", edgecolor="black", label="best")

        ax.set_xlabel(f"{names[0]} normalized")
        ax.set_ylabel(f"{names[1]} normalized")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("QL trajectory over selected surface")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(color="white", alpha=0.25)
        fig.tight_layout()
        out = mo.as_html(fig)
        plt.close(fig)
        out
    return


@app.cell
def _(history_audio_loss, history_p_loss, mo, np, plt, trajectory):
    if trajectory.size == 0:
        mo.md("")
    else:
        x = np.arange(len(history_p_loss))
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

        axes[0].plot(x, history_p_loss, color="#1f77b4")
        axes[0].set_title("P-Loss per evaluation")
        axes[0].set_xlabel("Evaluation")
        axes[0].set_ylabel("P-Loss")
        axes[0].grid(alpha=0.3)

        axes[1].plot(x, history_audio_loss, color="#d62728")
        axes[1].set_title("Audio loss per evaluation")
        axes[1].set_xlabel("Evaluation")
        axes[1].set_ylabel("Audio loss")
        axes[1].grid(alpha=0.3)

        fig.tight_layout()
        out = mo.as_html(fig)
        plt.close(fig)
        out
    return


@app.cell
def _(history_audio_loss, history_p_loss, mo, pd, trajectory):
    if trajectory.size == 0:
        mo.md("")
    else:
        rows = []
        for i, point in enumerate(trajectory):
            rows.append(
                {
                    "eval": i,
                    "x_norm": round(float(point[0]), 4),
                    "y_norm": round(float(point[1]), 4),
                    "p_loss": round(float(history_p_loss[i]), 5),
                    "audio_loss": round(float(history_audio_loss[i]), 5),
                }
            )
        mo.ui.table(pd.DataFrame(rows), selection=None)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
