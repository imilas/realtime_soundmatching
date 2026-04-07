"""
agent/features.py

Audio feature extraction for the loss function.

We use a log-magnitude power spectrum averaged over several blocks.
This is intentionally simple and swappable — you could replace
compute_features() with MFCCs, mel spectrograms, perceptual weighting, etc.
"""

import numpy as np
from scipy.signal import welch


# ------------------------------------------------------------------
# Core feature: log power spectral density
# ------------------------------------------------------------------

def compute_features(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    freq_range: tuple = (20, 8000),
) -> np.ndarray:
    """
    Compute a log-magnitude power spectrum (PSD via Welch's method).

    Parameters
    ----------
    audio       : 1-D float32 array of PCM samples
    sample_rate : sample rate in Hz
    n_fft       : FFT window size
    freq_range  : (low, high) Hz — trim spectrum to this band

    Returns
    -------
    log_psd : 1-D float64 array (same length for all calls with same params)
    """
    freqs, psd = welch(
        audio.astype(np.float64),
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft // 2,
        window="hann",
    )

    # Trim to audible band of interest
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    psd = psd[mask]

    # Log-compress; add floor to avoid log(0)
    log_psd = np.log1p(psd * 1e6)

    # L2-normalise so overall volume doesn't dominate
    norm = np.linalg.norm(log_psd)
    if norm > 0:
        log_psd /= norm

    return log_psd


# ------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------

def spectral_loss(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """L2 distance between two normalised log-PSD vectors."""
    return float(np.linalg.norm(features_a - features_b))


def spectral_loss_cosine(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """1 - cosine similarity (range [0, 2])."""
    dot = np.dot(features_a, features_b)
    denom = np.linalg.norm(features_a) * np.linalg.norm(features_b) + 1e-12
    return float(1.0 - dot / denom)


# ------------------------------------------------------------------
# Target loading
# ------------------------------------------------------------------

def load_target_from_wav(wav_path: str, sample_rate: int, **kwargs) -> np.ndarray:
    """Load a WAV file and return its feature vector."""
    import soundfile as sf
    audio, sr = sf.read(wav_path, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)      # stereo → mono
    if sr != sample_rate:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sample_rate, sr)
        audio = resample_poly(audio, sample_rate // g, sr // g)
    return compute_features(audio, sample_rate, **kwargs)


def load_target_from_capture(capture, n_blocks: int = 32, sample_rate: int = 44100, **kwargs) -> np.ndarray:
    """Capture n_blocks of audio from the target JACK port and return features."""
    print(f"[features] Recording {n_blocks} blocks for target…")
    audio = capture.get_n_blocks(n_blocks)
    return compute_features(audio, sample_rate, **kwargs)
