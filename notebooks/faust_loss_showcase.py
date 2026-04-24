import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from utils.loss_functions import ALL_LOSSES
    from utils.faust_renderer import render
    from utils.plotting import plot_spectrogram

    return ALL_LOSSES, mo, np, plot_spectrogram, plt, render


@app.cell
def _(mo):
    mo.md("""
    # Faust Loss Landscape Explorer

    Compare loss functions by sweeping a single parameter of an **imitator** synth
    against a fixed **target** sound, both rendered offline from Faust DSP programs.
    """)
    return


@app.cell
def _():
    # ── Configuration ──────────────────────────────────────────────
    target_dsp = "synths/bandpass_noise.dsp"
    target_params = {"hp_freq": 200, "lp_freq": 1000}

    imitator_dsp = "synths/bandpass_noise.dsp"
    imitator_fixed_params = {"hp_freq": 200}  # params held constant during sweep
    sweep_param = "lp_freq"                     # param to sweep
    sweep_min = 20
    sweep_max = 1500
    sweep_steps = 100

    duration_s = 1.0
    sample_rate = 44100
    return (
        duration_s,
        imitator_dsp,
        imitator_fixed_params,
        sample_rate,
        sweep_max,
        sweep_min,
        sweep_param,
        sweep_steps,
        target_dsp,
        target_params,
    )


@app.cell
def _(
    duration_s,
    mo,
    np,
    plot_spectrogram,
    plt,
    render,
    sample_rate,
    target_dsp,
    target_params,
):
    target_audio = render(target_dsp, params=target_params, duration_s=duration_s, sample_rate=sample_rate)

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 3))
    _axes[0].plot(np.linspace(0, duration_s, len(target_audio)), target_audio, linewidth=0.5)
    _axes[0].set_title("Target waveform")
    _axes[0].set_xlabel("Time (s)")
    plot_spectrogram(target_audio, sample_rate, ax=_axes[1], title="Target spectrogram")
    _fig.tight_layout()

    mo.md(f"**Target**: `{target_dsp}` with params `{target_params}`")
    return (target_audio,)


@app.cell
def _(
    ALL_LOSSES,
    duration_s,
    imitator_dsp,
    imitator_fixed_params,
    mo,
    np,
    render,
    sample_rate,
    sweep_max,
    sweep_min,
    sweep_param,
    sweep_steps,
    target_audio,
):
    sweep_values = np.linspace(sweep_min, sweep_max, sweep_steps)

    # render imitator at each sweep value
    imitator_audios = []
    for val in sweep_values:
        p = {**imitator_fixed_params, sweep_param: float(val)}
        imitator_audios.append(render(imitator_dsp, params=p, duration_s=duration_s, sample_rate=sample_rate))

    # compute all losses
    loss_results = {}
    for name, loss_fn in ALL_LOSSES.items():
        loss_results[name] = [loss_fn(target_audio, aud, sample_rate) for aud in imitator_audios]

    mo.md(f"Swept **{sweep_param}** over [{sweep_min}, {sweep_max}] in {sweep_steps} steps")
    return loss_results, sweep_values


@app.cell
def _(ALL_LOSSES, loss_results, plt, sweep_param, sweep_values, target_params):
    _n = len(ALL_LOSSES)
    _cols = 3
    _rows = (_n + _cols - 1) // _cols
    _fig, _axes = plt.subplots(_rows, _cols, figsize=(16, 4 * _rows))
    _axes = _axes.flatten()

    _target_val = target_params.get(sweep_param)

    for _i, (_name, _values) in enumerate(loss_results.items()):
        _ax = _axes[_i]
        _ax.plot(sweep_values, _values, linewidth=1.5)
        if _target_val is not None:
            _ax.axvline(_target_val, color="red", linestyle="dashed", alpha=0.6, label="target")
        _ax.set_title(_name, fontsize=14)
        _ax.set_xlabel(sweep_param)
        _ax.set_ylabel("Loss")
        _ax.legend(fontsize=9)

    for _j in range(_i + 1, len(_axes)):
        _axes[_j].set_visible(False)

    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
