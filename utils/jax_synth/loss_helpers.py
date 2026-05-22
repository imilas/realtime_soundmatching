"""
JAX-differentiable loss helpers ported from audio_nexting.

Provides:
  - spec_func(nfft, win_len, hop_len) -> spectrogram function
  - onset_1d(audio, kernel, spec_fn) -> 1-D onset envelope
  - gaussian_kernel1d(sigma, order, radius) -> kernel for onset smoothing
  - naive_loss(a, b) -> mean abs difference
  - clip_spec(x) -> spectrogram clipped to [0, 1]
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from audax.core import functional


def naive_loss(a, b):
    return jnp.abs(a - b).mean()


def clip_spec(x):
    return jnp.clip(x, 0.0, 1.0)


def gaussian_kernel1d(sigma: float, order: int, radius: int) -> np.ndarray:
    """1-D Gaussian (or derivative) kernel. Copied from scipy.ndimage._filters."""
    if order < 0:
        raise ValueError("order must be non-negative")
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()
    if order == 0:
        return phi_x
    q = np.zeros(order + 1)
    q[0] = 1
    D = np.diag(exponent_range[1:], 1)
    P = np.diag(np.ones(order) / -sigma2, -1)
    Q_deriv = D + P
    for _ in range(order):
        q = Q_deriv.dot(q)
    q = (x[:, None] ** exponent_range).dot(q)
    return q * phi_x


@partial(jax.jit, static_argnames=["sf"])
def onset_1d(target, k, sf):
    ts = sf(target)[0].sum(axis=1)
    onsets = jnp.convolve(ts, k, mode="same")
    return onsets


def spec_func(nfft: int, win_len: int, hop_len: int):
    """Return a JAX spectrogram function with the given STFT parameters."""
    window = jnp.hanning(win_len)
    return partial(
        functional.spectrogram,
        pad=0,
        window=window,
        n_fft=nfft,
        hop_length=hop_len,
        win_length=win_len,
        power=1,
        normalized=True,
        center=True,
        onesided=True,
    )
