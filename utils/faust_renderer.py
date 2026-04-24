"""
utils/faust_renderer.py

Offline rendering of Faust DSP programs to numpy arrays.
Uses faust2sndfile under the hood — no JACK required.
"""

import os
import subprocess
import tempfile
import hashlib
import numpy as np
import soundfile as sf


_COMPILE_CACHE_DIR = os.path.join(tempfile.gettempdir(), "faust_render_cache")


def _cache_key(dsp_path):
    """Hash the DSP source so we recompile only when it changes."""
    with open(dsp_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def compile_dsp(dsp_path):
    """
    Compile a .dsp file with faust2sndfile.

    Returns the path to the compiled binary.
    Caches in a temp directory — only recompiles if the source changes.
    """
    dsp_path = os.path.abspath(dsp_path)
    name = os.path.splitext(os.path.basename(dsp_path))[0]
    key = _cache_key(dsp_path)

    os.makedirs(_COMPILE_CACHE_DIR, exist_ok=True)
    binary = os.path.join(_COMPILE_CACHE_DIR, f"{name}_{key}")

    if os.path.isfile(binary):
        return binary

    # faust2sndfile wants the DSP in the working directory
    work_dir = os.path.dirname(dsp_path)
    result = subprocess.run(
        ["faust2sndfile", dsp_path],
        cwd=work_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"faust2sndfile failed:\n{result.stderr}")

    # faust2sndfile puts the binary next to the DSP file
    src_binary = os.path.join(work_dir, name)
    if not os.path.isfile(src_binary):
        raise FileNotFoundError(f"Expected binary at {src_binary}")

    import shutil
    shutil.move(src_binary, binary)
    return binary


def render(dsp_path, params=None, duration_s=1.0, sample_rate=44100):
    """
    Render a Faust DSP program to a numpy array.

    Parameters
    ----------
    dsp_path    : path to a .dsp file
    params      : dict of {param_name: value} to override defaults
    duration_s  : duration in seconds
    sample_rate : sample rate in Hz

    Returns
    -------
    audio : 1-D numpy float64 array (mono)
    """
    binary = compile_dsp(dsp_path)
    n_samples = int(duration_s * sample_rate)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        cmd = [binary, "-s", str(n_samples), "-sr", str(sample_rate)]
        if params:
            for name, value in params.items():
                # Faust CLI strips underscores/spaces from param names
                cli_name = name.replace("_", "").replace(" ", "")
                cmd.extend([f"-{cli_name}", str(value)])
        cmd.append(wav_path)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Render failed:\n{result.stderr}")

        audio, sr = sf.read(wav_path)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)  # stereo → mono
        return audio.astype(np.float64)
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
