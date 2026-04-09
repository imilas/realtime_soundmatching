import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import stft

    return mo, np, plt, stft


@app.cell
def _(mo):
    mo.md("""
    # Synth Experiment Notebook
    """)
    return


@app.cell
def _():
    synth_json  = "synths/bandpass_noise.dsp.json"
    jack_port   = "bandpass_noise:out_0"
    target_wav  = "targets/bp_900_3.wav"   # swap for a real wav
    sample_rate = 44100
    osc_host    = "127.0.0.1"
    osc_port    = 5510
    eval_blocks = 8
    return (
        eval_blocks,
        jack_port,
        osc_host,
        osc_port,
        sample_rate,
        synth_json,
        target_wav,
    )


@app.cell
def _(synth_json):
    from agent.params import FaustParams
    params = FaustParams(synth_json)
    return (params,)


@app.cell
def _(np, stft):
    def compute_features(audio, sample_rate, n_fft=2048, hop=512, freq_range=(20, 8000)):
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

    return compute_features, loss_fn


@app.cell
def _(compute_features, mo, sample_rate, target_wav):
    import soundfile as sf

    try:
        _audio, _sr = sf.read(target_wav, always_2d=False)
        if _audio.ndim == 2:
            _audio = _audio.mean(axis=1)
        target_features = compute_features(_audio, sample_rate)
        target_status = mo.callout(mo.md(f"Loaded `{target_wav}`  —  {len(_audio)} samples"), kind="success")
    except Exception as e:
        target_features = None
        target_status = mo.callout(mo.md(f"Could not load target: {e}"), kind="warn")

    target_status
    return (target_features,)


@app.cell
def _(mo, osc_host, osc_port, params):
    from agent.controller import OSCController
    controller = OSCController(params, host=osc_host, port=osc_port)

    sliders = {
        name: mo.ui.slider(
            start=p.min_val, stop=p.max_val, value=p.default,
            step=p.step, label=name, show_value=True,
        )
        for name, p in params.items()
    }



    mo.vstack([mo.md("## OSC Controls"), *sliders.values()])
    return controller, sliders


@app.cell
def _(controller, sliders):
    # Reactive send: re-runs whenever any slider moves
    controller.send({name: s.value for name, s in sliders.items()})
    return


@app.cell
def _(mo):
    capture_btn = mo.ui.run_button(label="Capture & Evaluate")
    capture_btn
    return (capture_btn,)


@app.cell
def _(
    JackCapture,
    capture_btn,
    compute_features,
    eval_blocks,
    jack_port,
    loss_fn,
    mo,
    sample_rate,
    target_features,
):
    capture_btn  # re-run on click

    _result = mo.md("_press button to capture_")

    if capture_btn.value:
        try:
            # from agent.capture import JackCapture
            _cap = JackCapture("nb_capture")
            _cap.start(jack_port)
            _audio = _cap.get_n_blocks(eval_blocks)
            _cap.stop()

            _feats = compute_features(_audio, sample_rate)
            _loss  = loss_fn(_feats, target_features) if target_features is not None else float("nan")
            _result = mo.callout(mo.md(f"**loss = {_loss:.5f}**"), kind="info")
        except Exception as e:
            _result = mo.callout(mo.md(f"Capture failed: {e}"), kind="danger")

    _result
    return


@app.cell
def _(
    capture_btn,
    compute_features,
    eval_blocks,
    jack_port,
    mo,
    plt,
    sample_rate,
    target_features,
):
    capture_btn  # refresh on each capture

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
    _axes[0].set_title("target")
    _axes[1].set_title("synth (last capture)")

    if target_features is not None:
        _axes[0].plot(target_features)

    if capture_btn.value:
        try:
            from agent.capture import JackCapture
            _cap = JackCapture("nb_capture_plot")
            _cap.start(jack_port)
            _audio = _cap.get_n_blocks(eval_blocks)
            _cap.stop()
            _feats = compute_features(_audio, sample_rate)
            _axes[1].plot(_feats, color="orange")
        except Exception:
            pass

    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return (JackCapture,)


@app.cell
def _(params):
    def step(current_vec, spectrogram, loss):
        """
        Return a new parameter vector given the current spectrogram and loss.
        current_vec : np.ndarray  — current param values
        spectrogram : np.ndarray  — output of compute_features()
        loss        : float
        """
        # TODO: your update rule here
        return current_vec  # no-op

    _vec = params.defaults_vector()
    print("step() ready, param dim =", len(_vec))
    return


if __name__ == "__main__":
    app.run()
