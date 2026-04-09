"""
agent/features.py

Audio feature extraction for the loss function.

We use a mean log-magnitude spectrogram (STFT-based) averaged over time frames.
This is intentionally simple and swappable — you could replace
compute_features() with MFCCs, mel spectrograms, perceptual weighting, etc.
"""

import numpy as np
from scipy.signal import stft


# ------------------------------------------------------------------
# Core feature: mean log-magnitude spectrogram
# ------------------------------------------------------------------

def compute_features(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    freq_range: tuple = (20, 8000),
) -> np.ndarray:
    """
    Compute a mean log-magnitude spectrogram via STFT.

    Each STFT frame gives a magnitude spectrum; we log-compress each frame,
    then average across time to produce a single 1-D feature vector.
    The result is L2-normalised so overall volume doesn't dominate.

    Parameters
    ----------
    audio       : 1-D float32 array of PCM samples
    sample_rate : sample rate in Hz
    n_fft       : FFT window size
    hop_length  : hop between successive frames
    freq_range  : (low, high) Hz — trim spectrum to this band

    Returns
    -------
    features : 1-D float64 array (same length for all calls with same params)
    """
    freqs, _, Zxx = stft(
        audio.astype(np.float64),
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
    )

    mag = np.abs(Zxx)  # shape: (n_freq_bins, n_frames)

    # Trim to audible band of interest
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    mag = mag[mask, :]

    # Log-compress per frame
    log_mag = np.log1p(mag * 1e3)

    # Average over time frames → 1-D
    features = log_mag.mean(axis=1)

    # L2-normalise
    norm = np.linalg.norm(features)
    if norm > 0:
        features /= norm

    return features


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
