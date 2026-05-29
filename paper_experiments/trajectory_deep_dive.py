import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

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
    )

    RESULTS_DIR = Path("paper_experiments/results")
    return (
        METHOD_ORDER,
        RESULTS_DIR,
        build_trial_context,
        compute_surface,
        extract_trajectory,
        load_all_results,
        make_eval_table,
        mo,
        plot_full_trajectory,
        plot_loss_curves,
        plot_step_trajectory,
        plt,
        trial_summary_md,
    )


@app.cell
def _(mo):
    _out = mo.md(
        """
        # Trajectory deep dive

        Step through a single optimization run for **any method** (GD, HillClimber,
        RandomSearch, CMA-ES, BO, QL) and watch the parameter updates move across the
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
        synth_picker = trial_picker = surface_picker = grid_slider = None
    else:
        synth_picker = mo.ui.dropdown(options=synths, value=synths[0], label="Synth")
        trial_picker = mo.ui.number(start=0, stop=10_000, step=1, value=0, label="Trial index")
        surface_picker = mo.ui.dropdown(
            options=["P-Loss surface", "Audio-loss surface"],
            value="P-Loss surface", label="Surface",
        )
        grid_slider = mo.ui.slider(start=15, stop=61, step=2, value=31, label="Grid resolution")
    return grid_slider, surface_picker, synth_picker, trial_picker


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
def _(
    grid_slider,
    method_picker,
    mo,
    surface_picker,
    synth_picker,
    trial_picker,
):
    if synth_picker is None:
        _out = mo.md("No result pickle files found in `paper_experiments/results`.")
    else:
        _out = mo.hstack([synth_picker, method_picker, trial_picker, surface_picker, grid_slider])
    _out
    return


@app.cell
def _(
    all_results,
    build_trial_context,
    extract_trajectory,
    method_picker,
    synth_picker,
    trial_picker,
):
    ctx = build_trial_context(all_results, synth_picker.value, method_picker.value, trial_picker.value)
    traj = extract_trajectory(ctx)
    return ctx, traj


@app.cell
def _(ctx, mo, traj, trial_summary_md):
    _out = mo.md(trial_summary_md(ctx, traj)) if ctx["trial"] else mo.md("No selected trial."
                                                                        )
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
    plot_full_trajectory,
    plt,
    surface,
    surface_label,
    traj,
    xx,
    yy,
):
    trajectory = traj["trajectory"]

    if trajectory.size == 0:
        _out = mo.md("This trial has no `history_params`.")
    elif trajectory.shape[1] != 2:
        _out = mo.md("Visualization only supports 2-parameter synths.")
    else:
        _fig = plot_full_trajectory(traj, ctx, xx, yy, surface, surface_label)
        _out = mo.as_html(_fig)
        plt.close(_fig)
    _out
    return (trajectory,)


@app.cell
def _(mo):
    _out = mo.md(
        """
        ### Step through the trajectory

        Drag the slider to reveal the first *k* evaluations one at a time.
        """
    )
    _out
    return


@app.cell
def _(mo, traj, trajectory):
    _trajectory = traj["trajectory"]
    if _trajectory.size == 0 or _trajectory.shape[1] != 2:
        step_slider = None
        _out = mo.md("")
    else:
        step_slider = mo.ui.slider(
            start=1, stop=len(trajectory), step=1,
            value=len(trajectory), label="Reveal first k evals",
        )
        _out = step_slider
    _out
    return (step_slider,)


@app.cell
def _(
    ctx,
    mo,
    plot_step_trajectory,
    plt,
    step_slider,
    surface,
    surface_label,
    traj,
    trajectory,
    xx,
    yy,
):
    if trajectory.size == 0 or trajectory.shape[1] != 2 or step_slider is None:
        _out = mo.md("")
    else:
        _fig = plot_step_trajectory(traj, ctx, xx, yy, surface, surface_label, int(step_slider.value))
        _out = mo.as_html(_fig)
        plt.close(_fig)
    _out
    return


@app.cell
def _(mo, plot_loss_curves, plt, step_slider, traj, trajectory):
    if trajectory.size == 0:
        _out = mo.md("")
    else:
        _k = int(step_slider.value) if step_slider is not None else len(traj["history_p_loss"])
        _fig = plot_loss_curves(traj, _k)
        _out = mo.as_html(_fig)
        plt.close(_fig)
    _out
    return


@app.cell
def _(make_eval_table, mo, traj, trajectory):
    if trajectory.size == 0:
        _out = mo.md("")
    
    else:
        _out = mo.ui.table(make_eval_table(traj), selection=None)
    _out
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
