"""
utils/plotting.py

Audio visualisation helpers.
"""

import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt


def compute_spectrogram(audio, sample_rate, n_fft=2048, hop_length=512,
                        freq_range=None, log=True):
    """
    Compute a magnitude spectrogram from a 1-D audio array.

    Parameters
    ----------
    audio       : 1-D float array
    sample_rate : sample rate in Hz
    n_fft       : FFT window size
    hop_length  : hop between frames
    freq_range  : optional (low_hz, high_hz) to crop frequency axis
    log         : if True, return log1p-compressed magnitudes

    Returns
    -------
    freqs  : 1-D array of frequency bin centres (Hz)
    times  : 1-D array of frame times (seconds)
    spec   : 2-D array (n_freq_bins, n_frames)
    """
    freqs, times, Zxx = stft(
        audio.astype(np.float64),
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
    )
    spec = np.abs(Zxx)

    if freq_range is not None:
        lo, hi = freq_range
        mask = (freqs >= lo) & (freqs <= hi)
        freqs = freqs[mask]
        spec = spec[mask, :]

    if log:
        spec = np.log1p(spec * 1e3)

    return freqs, times, spec


def plot_spectrogram(audio, sample_rate, n_fft=2048, hop_length=512,
                     freq_range=None, log=True, ax=None, title=None,
                     cmap="viridis"):
    """
    Compute and draw a spectrogram.

    Parameters
    ----------
    audio       : 1-D float array
    sample_rate : sample rate in Hz
    n_fft, hop_length, freq_range, log : passed to compute_spectrogram
    ax          : matplotlib Axes (created if None)
    title       : optional plot title
    cmap        : colormap name

    Returns
    -------
    ax : the matplotlib Axes with the spectrogram drawn
    """
    freqs, times, spec = compute_spectrogram(
        audio, sample_rate,
        n_fft=n_fft, hop_length=hop_length,
        freq_range=freq_range, log=log,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.pcolormesh(times, freqs, spec, shading="auto", cmap=cmap)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if title:
        ax.set_title(title)

    return ax
