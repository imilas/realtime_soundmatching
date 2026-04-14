import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import stft
    from agent.params import FaustParams
    import soundfile as sf
    import utils.io as io_utils
    import utils.marimo_helpers as mo_help



    return FaustParams, io_utils, mo, mo_help, np, stft


@app.cell
def _(mo):
    mo.md("""
    # Synth Experiment Notebook
    """)
    return


@app.cell
def _():
    synth_json  = "synths/sine.dsp.json"
    jack_port   = "sine:out_0"
    target_wav  = "targets/50hz_sine.wav"
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
def _(FaustParams, synth_json):
    params = FaustParams(synth_json)
    return (params,)


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

    return loss_fn, spec_features


@app.cell
def _(io_utils, mo, sample_rate, spec_features, target_wav):
    _audio, _sr = io_utils.load_audio(target_wav)
    target_features = spec_features(_audio, sample_rate)
    target_status = mo.callout(mo.md(f"Loaded `{target_wav}`  —  {len(_audio)} samples"), kind="success")
    target_status
    return (target_features,)


@app.cell
def _(mo, mo_help, osc_host, osc_port, params):
    from agent.controller import OSCController

    controller = OSCController(params, host=osc_host, port=osc_port)

    sliders = mo_help.make_slider(params)
 
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
    capture_btn,
    eval_blocks,
    jack_port,
    loss_fn,
    mo,
    sample_rate,
    spec_features,
    target_features,
):
    capture_btn  # re-run on click

    _result = mo.md("_press button to capture_")

    if capture_btn.value:
        try:
            from agent.capture import JackCapture
            _cap = JackCapture("nb_capture")
            _cap.start(jack_port)
            _audio = _cap.get_n_blocks(eval_blocks)
            _cap.stop()

            _feats = spec_features(_audio, sample_rate)
            _loss  = loss_fn(_feats, target_features) if target_features is not None else float("nan")
            _result = mo.callout(mo.md(f"**loss = {_loss:.5f}**"), kind="info")
        except Exception as e:
            _result = mo.callout(mo.md(f"Capture failed: {e}"), kind="danger")

    _result
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
