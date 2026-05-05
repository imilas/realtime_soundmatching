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

    return FaustParams, JackCapture, io_utils, loss_fns, mo, np, stft


@app.cell
def _():
    # l2_spectral_loss(np.zeros([4096]), np.ones([4096]),44100)
    return


@app.cell
def _(mo):
    mo.md("""
    # Synth Experiment Notebook
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
def _(mo, params):
    param_state, set_param_state = mo.state(
        {name: p.default for name, p in params.items()}
    )
    sent_cache = {}  # tracks values sent by notebook sliders, used to filter echoes
    return param_state, sent_cache, set_param_state


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
def _(mo, osc_host, osc_port, param_state, params, sent_cache):
    from agent.controller import OSCController
    import utils.marimo_helpers as mo_help

    controller = OSCController(params, host=osc_host, port=osc_port)

    def auto_send_osc(name, value):
        sent_cache[name] = value
        controller.send({name: value})

    sliders = mo_help.make_slider(params, values=param_state(), on_change_func=auto_send_osc)
    mo.vstack([mo.md("## OSC Controls"), *sliders.values()])
    return


@app.cell
def _(mo):
    capture_btn = mo.ui.run_button(label="Capture & Evaluate")
    capture_btn
    return (capture_btn,)


@app.cell(hide_code=True)
def _(
    JackCapture,
    capture_btn,
    eval_blocks,
    jack_port,
    loss_fns,
    mo,
    target_audio,
):
    capture_btn  # re-run on click

    _result = mo.md("_press button to capture_")

    if capture_btn.value:
        try:

            _cap = JackCapture("nb_capture")
            try:
                _cap.start(jack_port)
                audio = _cap.get_n_blocks(eval_blocks)
                print(len(audio), len(target_audio))
                _loss = loss_fns.multi_resolution_spectral_loss(audio,target_audio,sample_rate=44100)
                _result = mo.callout(mo.md(f"**loss = {_loss:.5f}**"), kind="info")
                rec_audio = audio
            finally:
                _cap.stop()
        except Exception as e:
            _result = mo.callout(mo.md(f"Capture failed: {e}"), kind="danger")

    _result
    return (audio,)


@app.cell
def _():
    # loss_refresh = mo.ui.refresh(default_interval=2)
    # loss_refresh
    return


@app.cell
def _():
    # loss_refresh

    # try:
    #     _cap = JackCapture("nb_capture")
    #     try:
    #         _cap.start(jack_port)
    #         audio = _cap.get_n_blocks(eval_blocks)
    #         print(len(audio), len(target_audio))
    #         # _loss = loss_fns.ssim_spectral_loss(audio,target_audio,sample_rate=44100)
    #         _loss = loss_fns.dtw_onset_loss(audio,target_audio,sample_rate=44100)
    #         _result = mo.callout(mo.md(f"**loss = {_loss:.5f}**"), kind="info")
    #     finally:
    #         _cap.stop()
    # except Exception as e:
    #     _result = mo.callout(mo.md(f"Capture failed: {e}"), kind="danger")
    # _result
    return


@app.cell
def _(audio, mo, sample_rate, target_audio):
    mo.hstack([
        mo.vstack([
            mo.md("**Target**"),
            mo.audio(target_audio, rate=sample_rate),
        ]),
        mo.vstack([
            mo.md("**Output**"),
            mo.audio(audio, rate=sample_rate),
        ]),
    ])
    return


@app.cell
def _(audio):
    from utils.plotting import compute_spectrogram, plot_spectrogram

    # just the data
    freqs, times, spec = compute_spectrogram(audio, 44100, freq_range=(20, 5000))

    # plot it
    plot_spectrogram(audio, 44100, title="My signal", freq_range=(20, 5000))
    return (plot_spectrogram,)


@app.cell
def _(plot_spectrogram, target_audio):
    plot_spectrogram(target_audio, 44100, title="My signal", freq_range=(20, 5000))
    return


if __name__ == "__main__":
    app.run()
