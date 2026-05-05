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
    from scipy.signal import stft
    from agent.params import FaustParams
    from agent.capture import JackCapture
    import soundfile as sf
    import utils.io as io_utils

    import utils.loss_functions as loss_fns

    return FaustParams, io_utils, loss_fns, mo, np, plt, stft


@app.cell
def _(mo):
    mo.md("""
    # 1-Parameter Agent Demo

    **Goal**: Visualize how a random search agent explores a single parameter of a synthesizer.

    - **Target**: Pre-recorded sound (`bp_100-901.wav` — bandpass noise with hp_freq=100, lp_freq=901)
    - **Imitator**: Same synth, but only varying `lp_freq` (the sweep parameter)
    - **Loss landscape**: Offline computation showing loss vs. each possible lp_freq value
    - **Agent**: Random search that steps ±X% on lp_freq, shown live on the landscape
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    # Synth Experiment Notebook
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Configuration
    """)
    return


@app.cell
def _():
    synth_json  = "synths/bandpass_noise.dsp.json"
    jack_port = "bandpass_noise:out_0"
    # jack_port   = "sine:out_0"
    target_wav  = "targets/bp_100-901.wav"
    sample_rate = 44100
    osc_host    = "127.0.0.1"
    osc_port    = 5510
    eval_blocks = 32 # block length is set in jack, i'm using 1024
    block_len = 1024 # this is set in jack
    return (
        block_len,
        eval_blocks,
        osc_host,
        osc_port,
        sample_rate,
        synth_json,
        target_wav,
    )


@app.cell
def _(FaustParams, synth_json):
    params = FaustParams(synth_json)
    return (params,)


@app.cell
def _(mo, params):
    param_state, set_param_state = mo.state(
        {name: p.default for name, p in params.items()}
    )
    sent_cache = {}  # tracks values sent by notebook sliders, used to filter echoes
    agent_history, set_agent_history = mo.state([])
    return (
        agent_history,
        param_state,
        sent_cache,
        set_agent_history,
        set_param_state,
    )


@app.cell
def _(osc_port, params, sent_cache, set_param_state):
    from pythonosc import dispatcher as _dispatcher, osc_server as _osc_server
    import threading as _threading

    _disp = _dispatcher.Dispatcher()

    def _on_param(address, value):
        for name, p in params.items():
            if p.osc_address == address:
                # skip echoes of values we just sent from the notebook
                if name in sent_cache and abs(sent_cache.pop(name) - float(value)) < 1.0:
                    return
                set_param_state(lambda s, n=name, v=float(value): {**s, n: v})
                break

    for _name, _p in params.items():
        _disp.map(_p.osc_address, _on_param)

    try:
        _server = _osc_server.ThreadingOSCUDPServer(("127.0.0.1", osc_port + 1), _disp)
        _threading.Thread(target=_server.serve_forever, daemon=True).start()
    except OSError:
        print(f"[OSC Listener] Could not bind to port {osc_port + 1}")
    return


@app.cell
def _(np, stft):
    def spec_features(audio, sample_rate, n_fft=2048, hop=512, freq_range=(20, 5000)):
        freqs, _, Zxx = stft(audio.astype(np.float64), fs=sample_rate,
                             window="hann", nperseg=n_fft, noverlap=n_fft - hop)
        mag = np.abs(Zxx)
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        log_mag = np.log1p(mag[mask] * 1e3)
        feat = log_mag.mean(axis=1)
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat

    def loss_fn(a, b):
        """L2 distance between two feature vectors."""
        return float(np.linalg.norm(a - b))

    return


@app.cell
def _():
    # loss_fn(np.zeros([4096]), np.ones([4096]))
    return


@app.cell
def _(block_len, eval_blocks, io_utils, target_wav):
    target_audio, _sr = io_utils.load_audio(target_wav)
    target_audio = target_audio[:eval_blocks * block_len]  # trim to eval length
    return (target_audio,)


@app.cell
def _(mo):
    mo.md("""
    ## Loss Landscape Setup

    Pre-compute loss as a function of a single parameter. The agent will try to minimize loss by adjusting this parameter.
    """)
    return


@app.cell
def _():
    # 1-param sweep config: which parameter to optimize, what to hold fixed
    sweep_param   = "lp_freq"
    fixed_params  = {"hp_freq": 100}   # held constant; matches bp_100-901.wav target
    sweep_min     = 20.0
    sweep_max     = 8000.0
    sweep_steps   = 80
    return fixed_params, sweep_max, sweep_min, sweep_param, sweep_steps


@app.cell
def _(
    fixed_params,
    loss_fns,
    mo,
    np,
    sample_rate,
    sweep_max,
    sweep_min,
    sweep_param,
    sweep_steps,
    target_audio,
):
    from utils.faust_renderer import render as _render

    # Render imitator synth at each sweep value and compute loss vs target
    _sweep_values = np.linspace(sweep_min, sweep_max, sweep_steps)
    _landscape = []
    for _v in _sweep_values:
        # Render with sweep param at current value, all other params fixed
        _audio = _render(
            "synths/bandpass_noise.dsp",
            params={**fixed_params, sweep_param: float(_v)},
            duration_s=len(target_audio) / sample_rate,
            sample_rate=sample_rate
        )
        _audio = _audio[:len(target_audio)]  # trim to match target length
        # Compute loss: how different is this from the target sound?
        _landscape.append(loss_fns.dtw_onset_loss(target_audio, _audio, sample_rate=sample_rate))

    landscape_x = _sweep_values
    landscape_y = np.array(_landscape)
    mo.md(f"Loss landscape computed over {sweep_steps} points")
    return landscape_x, landscape_y


@app.cell
def _(mo, osc_host, osc_port, param_state, params, sent_cache):
    from agent.controller import OSCController
    import utils.marimo_helpers as mo_help

    controller = OSCController(params, host=osc_host, port=osc_port)

    def auto_send_osc(name, value):
        sent_cache[name] = value
        controller.send({name: value})

    sliders = mo_help.make_slider(params, values=param_state(), on_change_func=auto_send_osc)
    mo.vstack([mo.md("## OSC Controls"), *sliders.values()])
    return (controller,)


@app.cell
def _(mo):
    mo.md("""
    ## Live Landscape Visualization

    **Blue curve**: Loss landscape — the ground truth showing how loss varies with the parameter

    **Red star**: Current agent position — where it is right now

    **Orange dots**: Exploration trail — the path the agent took to get here (last ~40 steps)
    """)
    return


@app.cell
def _(
    agent_history,
    landscape_x,
    landscape_y,
    np,
    param_state,
    plt,
    sweep_param,
):
    _fig, _ax = plt.subplots(figsize=(10, 4))

    # Plot the pre-computed loss landscape
    _ax.plot(landscape_x, landscape_y, linewidth=2, label="Loss landscape", color="blue")

    # Trail of past positions (last 40)
    _hist = agent_history()
    if _hist:
        _hist_vals = np.array(_hist[-40:])
        _hist_losses = np.interp(_hist_vals, landscape_x, landscape_y)
        _ax.scatter(_hist_vals, _hist_losses, alpha=0.25, s=20, color="orange", zorder=4)

    # Current position
    _current_val = param_state()[sweep_param]
    _idx = np.argmin(np.abs(landscape_x - _current_val))
    _ax.scatter([_current_val], [landscape_y[_idx]], color="red", s=150, zorder=5, marker="*", label="current")

    _ax.set_xlabel(sweep_param)
    _ax.set_ylabel("Loss")
    _ax.set_title(f"Loss landscape & agent position")
    _ax.legend(fontsize=10)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Random Search Agent

    The agent randomly changes the parameter each step, without using gradient information.
    Check the box to enable, adjust the step size slider to control exploration speed.
    """)
    return


@app.cell
def _(mo):
    agent_enabled = mo.ui.checkbox(label="▶ / ■  Random agent")
    step_pct = mo.ui.slider(1, 20, value=5, label="Step size (% of range)")
    agent_refresh = mo.ui.refresh(default_interval=1)
    mo.hstack([agent_enabled, step_pct, agent_refresh])
    return agent_enabled, agent_refresh, step_pct


@app.cell
def _(
    agent_enabled,
    agent_refresh,
    controller,
    np,
    param_state,
    params,
    sent_cache,
    set_agent_history,
    set_param_state,
    step_pct,
    sweep_param,
):
    agent_refresh

    if agent_enabled.value:
        # Get current parameter value
        _current = param_state()
        _p = params[sweep_param]

        # Random direction: +1 or -1
        _direction = np.random.choice([-1, 1])

        # Step size as percentage of parameter range
        _step = (_p.max_val - _p.min_val) * (step_pct.value / 100)

        # Compute new value and clamp to bounds
        _new_val = float(np.clip(
            _current[sweep_param] + _direction * _step,
            _p.min_val,
            _p.max_val
        ))

        # Send new parameter value to synth via OSC
        sent_cache[sweep_param] = _new_val
        controller.send({sweep_param: _new_val})

        # Update state so UI and plots reflect the change
        set_param_state(lambda s, v=_new_val: {**s, sweep_param: v})
        set_agent_history(lambda h, v=_new_val: h + [v])
    return


if __name__ == "__main__":
    app.run()
