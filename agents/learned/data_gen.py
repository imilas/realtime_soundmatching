"""
Data generation + featurization for the learned amortized inverse model (E5 B1).

Samples params uniformly in normalized space, renders audio, and turns each
clip into a compact log-mel feature vector. Targets are the normalized params,
so a model trained on (features -> params) is a learned inverse synth.
"""
from __future__ import annotations

import numpy as np

from agent.params import FaustParams
from experiments.multidim_runner import _bounds_from_params, _render_audio
from synths.build import prepare

SAMPLE_RATE = 44100
N_SAMPLES = 44100  # 1 s
N_MELS = 64
N_TIME = 8         # coarse temporal pooling (keeps am modulation structure)
FEATURE_DIM = N_MELS * N_TIME


def _pool(mat: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """Block-mean pool `mat` along `axis` to exactly n_out bins."""
    n = mat.shape[axis]
    idx = np.linspace(0, n, n_out + 1).astype(int)
    sl = [slice(None)] * mat.ndim
    out = []
    for i in range(n_out):
        sl[axis] = slice(idx[i], max(idx[i] + 1, idx[i + 1]))
        out.append(mat[tuple(sl)].mean(axis=axis))
    return np.stack(out, axis=axis)


def featurize(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Log-magnitude STFT pooled to N_MELS x N_TIME, flattened + z-scored.

    Uses scipy STFT (no librosa/numba — those can't cache in the sandbox).
    """
    from scipy.signal import stft

    _f, _t, Zxx = stft(audio.astype(np.float64), fs=sample_rate,
                       nperseg=2048, noverlap=2048 - 512, window="hann")
    logmag = np.log1p(np.abs(Zxx))                 # (F, T)
    pooled = _pool(_pool(logmag, N_MELS, axis=0), N_TIME, axis=1)  # (N_MELS, N_TIME)
    feat = pooled.reshape(-1)
    return (feat - feat.mean()) / (feat.std() + 1e-8)


def synth_context(synth: str):
    """Return (build, params, bounds) for a synth (cached prepare)."""
    build = prepare(synth)
    params = FaustParams(str(build.json_path))
    bounds = _bounds_from_params(params)
    return build, params, bounds


def generate_dataset(synth: str, n: int, seed: int = 12345):
    """Return (X[n, FEATURE_DIM], Y[n, d]) of features and NORMALIZED params."""
    build, params, bounds = synth_context(synth)
    rng = np.random.default_rng(seed)
    X = np.empty((n, FEATURE_DIM), dtype=np.float32)
    Y = np.empty((n, bounds.d), dtype=np.float32)
    for i in range(n):
        u = rng.uniform(0.0, 1.0, size=bounds.d)
        real = bounds.denormalize(u)
        audio = _render_audio(str(build.dsp_path), params.vector_to_dict(real), N_SAMPLES, SAMPLE_RATE)
        X[i] = featurize(audio)
        Y[i] = u
        if (i + 1) % 1000 == 0:
            print(f"  [{synth}] generated {i + 1}/{n}", flush=True)
    return X, Y


def benchmark_targets(synth: str, n_seeds: int = 200):
    """Regenerate the EXACT targets the optimizers faced (run_trial's logic),
    returning (X[n, FEATURE_DIM], true_norm[n, d]) for held-out evaluation."""
    build, params, bounds = synth_context(synth)
    X = np.empty((n_seeds, FEATURE_DIM), dtype=np.float32)
    Y = np.empty((n_seeds, bounds.d), dtype=np.float32)
    for seed in range(n_seeds):
        sub = np.random.SeedSequence(seed).generate_state(2)
        rng = np.random.default_rng(sub[0])
        true_norm = rng.uniform(0.0, 1.0, size=bounds.d)  # matches run_trial
        real = bounds.denormalize(true_norm)
        audio = _render_audio(str(build.dsp_path), params.vector_to_dict(real), N_SAMPLES, SAMPLE_RATE)
        X[seed] = featurize(audio)
        Y[seed] = true_norm
    return X, Y
