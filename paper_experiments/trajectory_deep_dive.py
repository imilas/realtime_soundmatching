import marimo

__generated_with = "0.8.22"
app = marimo.App(width="full")


@app.cell
def _(__file__):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    from paper_experiments.config import AUDIO_DURATION_S, SAMPLE_RATE
    from utils.notebooks.trajectory_funcs import (
        METHOD_ORDER,
        load_all_results,
        build_trial_context,
        extract_trajectory,
        compute_surface,
        plot_full_trajectory,
        plot_step_trajectory,
        plot_loss_curves,
        make_eval_table,
        trial_summary_md,
        render_trial_audios,
        audio_to_wav_bytes,
        skip_indices,
        fill_dsp_params,
    )

    RESULTS_DIR = Path("paper_experiments/results")
    return (
        AUDIO_DURATION_S,
        METHOD_ORDER,
        Path,
        RESULTS_DIR,
        SAMPLE_RATE,
        audio_to_wav_bytes,
        build_trial_context,
        compute_surface,
        extract_trajectory,
        fill_dsp_params,
        load_all_results,
        make_eval_table,
        mo,
        np,
        pd,
        plot_full_trajectory,
        plot_loss_curves,
        plot_step_trajectory,
        plt,
        render_trial_audios,
        skip_indices,
        sys,
        trial_summary_md,
    )


@app.cell
def _(mo):
    _out = mo.md(
        """
        # Trajectory deep dive

        Step through a single optimization run for **any method** (GD, HillClimber,
        RandomSearch, CMA-ES, QL) and watch the parameter updates move across the
        2D loss surface.  Pick a synth, method, and trial below.
        """
    )
    _out
    return


@app.cell
def _(RESULTS_DIR, load_all_results):
    all_results = load_all_results(RESULTS_DIR)
    synths = sorted(all_results)
    return all_results, synths


@app.cell
def _(mo, synths):
    if not synths:
        synth_picker = trial_picker = surface_picker = grid_slider = skip_slider = None
    else:
        synth_picker = mo.ui.dropdown(options=synths, value=synths[0], label="Synth")
        trial_picker = mo.ui.number(start=0, stop=10_000, step=1, value=0, label="Trial index")
        surface_picker = mo.ui.dropdown(
            options=["P-Loss surface", "Audio-loss surface"],
            value="Audio-loss surface", label="Surface",
        )
        grid_slider = mo.ui.slider(start=2, stop=30, step=1, value=5, label="Grid resolution")
        skip_slider = mo.ui.slider(start=0, stop=90, step=5, value=20, label="Skip % of evals (keep first 10)")
    return (
        grid_slider,
        skip_slider,
        surface_picker,
        synth_picker,
        trial_picker,
    )


@app.cell
def _(METHOD_ORDER, all_results, mo, synth_picker):
    if synth_picker is None:
        method_picker = None
    else:
        _avail = all_results[synth_picker.value]
        _methods = [m for m in METHOD_ORDER if m in _avail]
        _methods += [m for m in sorted(_avail) if m not in _methods]
        method_picker = mo.ui.dropdown(options=_methods, value=_methods[0], label="Method")
    return (method_picker,)


@app.cell
def _(all_results, method_picker, mo, synth_picker):
    if synth_picker is None or method_picker is None:
        loss_picker = None
    else:
        _losses = sorted(all_results[synth_picker.value][method_picker.value].keys())
        loss_picker = mo.ui.dropdown(options=_losses, value=_losses[0], label="Loss")
    return (loss_picker,)


@app.cell
def _(
    loss_picker,
    method_picker,
    mo,
    skip_slider,
    synth_picker,
    trial_picker,
):
    if synth_picker is None:
        _out = mo.md("No result pickle files found in `paper_experiments/results`.")
    else:
        _out = mo.hstack([synth_picker, method_picker, loss_picker, trial_picker, skip_slider])
    _out
    return


@app.cell
def _(
    all_results,
    build_trial_context,
    extract_trajectory,
    fill_dsp_params,
    loss_picker,
    method_picker,
    mo,
    synth_picker,
    trial_picker,
):
    ctx = build_trial_context(
        all_results, synth_picker.value, method_picker.value, trial_picker.value,
        loss=loss_picker.value if loss_picker else None,
    )
    traj = extract_trajectory(ctx)
    _src = ctx['build'].dsp_path.read_text()
    _true = ctx['trial'].get('true_params', {}) if ctx['trial'] else {}
    _out = mo.vstack([
        mo.md("**With true parameters**"),
        mo.md(f"```faust\n{fill_dsp_params(_src, _true)}\n```"),
    ])
    _out
    return ctx, traj


@app.cell
def _(ctx, mo, traj, trial_summary_md):
    _out = mo.md(trial_summary_md(ctx, traj)) if ctx["trial"] else mo.md("No selected trial.")
    _out
    return


@app.cell
def _(mo, plot_loss_curves, plt, traj):
    if not traj["history_p_loss"]:
        _out = mo.md("")
    else:
        _fig = plot_loss_curves(traj, len(traj["history_p_loss"]))
        _out = mo.as_html(_fig)
        plt.close(_fig)
    _out
    return


@app.cell
def _(audio_to_wav_bytes, ctx, mo, render_trial_audios):
    _clips = render_trial_audios(ctx)
    if not _clips:
        _out = mo.md("No trial selected.")
    else:
        def _param_md(params):
            rows = "\n".join(f"| {k} | {v:.4f} |" for k, v in params.items())
            return mo.md(f"| param | value |\n|---|---|\n{rows}")
        _out = mo.vstack([
            mo.md("### Audio comparison"),
            mo.hstack([
                mo.vstack([mo.md(f"**{_label}**"), mo.audio(audio_to_wav_bytes(_audio)), _param_md(_params)])
                for _label, _audio, _params in _clips
            ]),
        ])
    _out
    return


@app.cell
def _():
    # _trajectory = traj["trajectory"]
    # if _trajectory.size == 0:
    #     _out = mo.md("")
    # else:
    #     _out = mo.ui.table(make_eval_table(traj), selection=None)
    # _out
    return


@app.cell
def _(mo):
    norm_picker = mo.ui.dropdown(
        options=["Linear", "Log", "Clip 95th pct"],
        value="Linear", label="Color scale",
    )
    norm_picker
    return (norm_picker,)


@app.cell
def _(grid_slider, mo, norm_picker, surface_picker):
    _out = mo.hstack([surface_picker, grid_slider, norm_picker])
    _out
    return


@app.cell
def _(compute_surface, ctx, grid_slider, surface_picker, traj):
    xx, yy, surface, surface_label = compute_surface(
        surface_picker.value, int(grid_slider.value), ctx, traj
    )
    return surface, surface_label, xx, yy


@app.cell
def _(
    ctx,
    mo,
    norm_picker,
    np,
    plot_full_trajectory,
    plt,
    skip_indices,
    skip_slider,
    surface,
    surface_label,
    traj,
    xx,
    yy,
):
    clip_pct = 10  # keep bottom 10% of values, clip the rest
    threshold = np.percentile(surface, clip_pct)
    surface_clipped = np.clip(surface, None, threshold) 

    trajectory = traj["trajectory"]
    if trajectory.size == 0:
        _out = mo.md("This trial has no `history_params`.")
    elif trajectory.shape[1] != 2:
        _out = mo.md("Visualization only supports 2-parameter synths.")
    else:
        _idx = skip_indices(len(trajectory), skip_slider.value if skip_slider else 0)
        _fig = plot_full_trajectory(traj, ctx, xx, yy, surface_clipped, surface_label,
                                    indices=_idx, norm_mode=norm_picker.value)
        _out = mo.as_html(_fig)
        plt.close(_fig)
    _out
    return clip_pct, surface_clipped, threshold, trajectory


@app.cell
def _(mo):
    mo.md("""### Step through the trajectory""")
    return


@app.cell
def _():
    # _trajectory = traj["trajectory"]
    # if _trajectory.size == 0 or _trajectory.shape[1] != 2:
    #     step_slider = None
    #     _out = mo.md("")
    # else:
    #     step_slider = mo.ui.slider(
    #         start=1, stop=len(trajectory), step=1,
    #         value=len(trajectory), label="Reveal first k evals",
    #     )
    #     _out = step_slider
    # _out
    return


@app.cell
def _():
    # if trajectory.size == 0 or trajectory.shape[1] != 2 or step_slider is None:
    #     _out = mo.md("")
    # else:
    #     _fig = plot_step_trajectory(traj, ctx, xx, yy, surface, surface_label, int(step_slider.value))
    #     _out = mo.as_html(_fig)
    #     plt.close(_fig)
    # _out
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
