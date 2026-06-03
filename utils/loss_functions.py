"""
utils/loss_functions.py

Loss functions for comparing two audio signals.
Every public function has the signature:
    loss(audio_a, audio_b, sample_rate) -> float
where audio_a and audio_b are 1-D numpy arrays.

Ported from JAX originals in utils/loss_helpers.py and
utils/loss_functions_definitions.py.
"""

import numpy as np
from scipy.signal import stft
from scipy.ndimage import gaussian_filter1d, uniform_filter

# Parameters matching the JAX GD pipeline (gradient_descent.py / loss_helpers.py)
_GD_NFFT    = 512
_GD_WIN_LEN = 600
_GD_HOP_LEN = 100


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _magnitude_spectrogram(audio, n_fft=2048, hop_length=512):
    """STFT → magnitude spectrogram (n_freq_bins, n_frames)."""
    _, _, Zxx = stft(
        audio.astype(np.float64),
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
    )
    return np.abs(Zxx)


def _log_mean_spectrum(audio, sample_rate, n_fft=2048, hop_length=512,
                       freq_range=(20, 8000)):
    """Log-magnitude spectrum averaged over time, L2-normalised."""
    freqs, _, Zxx = stft(
        audio.astype(np.float64),
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
    )
    mag = np.abs(Zxx)
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    log_mag = np.log1p(mag[mask, :] * 1e3)
    feat = log_mag.mean(axis=1)
    norm = np.linalg.norm(feat)
    return feat / norm if norm > 0 else feat


def _onset_envelope(audio, sample_rate, n_fft=1024, hop_length=256, sigma=3.0):
    """Onset strength envelope via spectral flux + Gaussian smoothing."""
    mag = _magnitude_spectrogram(audio, n_fft=n_fft, hop_length=hop_length)
    # spectral flux: positive first-order difference along time
    flux = np.diff(mag, axis=1)
    flux = np.maximum(flux, 0)
    # sum across frequency bins → 1-D envelope
    envelope = flux.sum(axis=0)
    # smooth
    envelope = gaussian_filter1d(envelope, sigma=sigma)
    return envelope


def _ssim_2d(img_a, img_b, win_size=7):
    """Structural similarity between two 2-D arrays."""
    img_a = img_a.astype(np.float64)
    img_b = img_b.astype(np.float64)
    data_range = max(img_a.max() - img_a.min(), img_b.max() - img_b.min())
    if data_range == 0:
        return 1.0

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu_a = uniform_filter(img_a, size=win_size)
    mu_b = uniform_filter(img_b, size=win_size)

    sigma_a_sq = uniform_filter(img_a ** 2, size=win_size) - mu_a ** 2
    sigma_b_sq = uniform_filter(img_b ** 2, size=win_size) - mu_b ** 2
    sigma_ab = uniform_filter(img_a * img_b, size=win_size) - mu_a * mu_b

    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a_sq + sigma_b_sq + C2)

    return float(np.mean(num / den))


def _dtw_distance(seq_a, seq_b, squared=False, normalize=True):
    """Dynamic time warping distance between two 1-D sequences.

    squared=True, normalize=False matches JAX SoftDTW(gamma=1) from softdtw_jax.py
    (which uses squared L2 cost and returns the raw accumulated distance).
    """
    n, m = len(seq_a), len(seq_b)
    if squared:
        cost = (seq_a[:, None] - seq_b[None, :]) ** 2
    else:
        cost = np.abs(seq_a[:, None] - seq_b[None, :])
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m) if normalize else D[n, m])


# ------------------------------------------------------------------
# Public loss functions
# ------------------------------------------------------------------

def l2_spectral_loss(audio_a, audio_b, sample_rate):
    """L2 distance between L2-normalised log-magnitude spectra."""
    feat_a = _log_mean_spectrum(audio_a, sample_rate)
    feat_b = _log_mean_spectrum(audio_b, sample_rate)
    return float(np.linalg.norm(feat_a - feat_b))


def multi_resolution_spectral_loss(audio_a, audio_b, sample_rate):
    """Sum of (L1 + log-L1) spectral losses at multiple FFT resolutions."""
    fft_sizes = [256, 512, 1024, 2048]
    eps = 1e-7
    total = 0.0
    for n_fft in fft_sizes:
        hop = n_fft // 4
        spec_a = _magnitude_spectrogram(audio_a, n_fft=n_fft, hop_length=hop)
        spec_b = _magnitude_spectrogram(audio_b, n_fft=n_fft, hop_length=hop)
        # trim to same shape
        min_t = min(spec_a.shape[1], spec_b.shape[1])
        spec_a, spec_b = spec_a[:, :min_t], spec_b[:, :min_t]
        l1 = np.mean(np.abs(spec_a - spec_b))
        log_l1 = np.mean(np.abs(np.log(spec_a + eps) - np.log(spec_b + eps)))
        total += l1 + log_l1
    return float(total)


def ssim_spectral_loss(audio_a, audio_b, sample_rate):
    """1 - SSIM between magnitude spectrograms (clipped to [0, 1])."""
    spec_a = _magnitude_spectrogram(audio_a, n_fft=1024, hop_length=256)
    spec_b = _magnitude_spectrogram(audio_b, n_fft=1024, hop_length=256)
    min_t = min(spec_a.shape[1], spec_b.shape[1])
    spec_a, spec_b = spec_a[:, :min_t], spec_b[:, :min_t]
    # normalise to [0, 1]
    peak = max(spec_a.max(), spec_b.max(), 1e-12)
    spec_a = np.clip(spec_a / peak, 0, 1)
    spec_b = np.clip(spec_b / peak, 0, 1)
    return float(1.0 - _ssim_2d(spec_a, spec_b))


def dtw_onset_loss(audio_a, audio_b, sample_rate):
    """DTW distance between onset envelopes."""
    env_a = _onset_envelope(audio_a, sample_rate)
    env_b = _onset_envelope(audio_b, sample_rate)
    return _dtw_distance(env_a, env_b)


# ------------------------------------------------------------------
# GD-matching helpers (same STFT params as JAX pipeline)
# ------------------------------------------------------------------

def _gd_spectrogram(audio: np.ndarray) -> np.ndarray:
    """
    Magnitude spectrogram matching audax/loss_helpers.spec_func exactly.
    Returns shape (n_frames, n_freq) = (time, freq).
    Params: nfft=512, win_len=600, hop=100, Hann window, reflect-pad by nfft//2,
    normalized by sqrt(sum(window^2)), power=1.
    """
    window = np.hanning(_GD_WIN_LEN)
    pad = _GD_NFFT // 2
    padded = np.pad(audio.astype(np.float32), pad, mode="reflect")
    n_frames = (len(padded) - _GD_WIN_LEN) // _GD_HOP_LEN + 1
    frames = np.stack([
        padded[i * _GD_HOP_LEN : i * _GD_HOP_LEN + _GD_WIN_LEN] * window
        for i in range(n_frames)
    ])  # (n_frames, win_len)
    # Truncate to nfft before FFT (win_len > nfft in the GD config)
    spec = np.abs(np.fft.rfft(frames[:, :_GD_NFFT], n=_GD_NFFT))  # (n_frames, n_freq)
    spec /= np.sqrt(np.sum(window ** 2))
    return spec


def _gd_onset_envelope(audio: np.ndarray) -> np.ndarray:
    """
    Onset envelope matching onset_1d in loss_helpers.py.
    Sum spectrogram over freq axis, then convolve with Gaussian(sigma=3, radius=10).
    """
    spec = _gd_spectrogram(audio)        # (n_frames, n_freq)
    energy = spec.sum(axis=1)            # (n_frames,) — total energy per frame
    kernel = _gaussian_kernel1d(sigma=3, radius=10)
    return np.convolve(energy, kernel, mode="same")


def _gaussian_kernel1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * x ** 2 / sigma ** 2)
    return (k / k.sum()).astype(np.float32)


def simse_spec_loss(audio_a: np.ndarray, audio_b: np.ndarray, sample_rate: int) -> float:
    """
    Scale-Invariant MSE on GD spectrograms, clipped to [0,1].
    Matches dm_pix.simse used in gradient_descent.py for bandpass_noise.
    alpha = dot(a, b) / dot(b, b);  loss = mean((a - alpha*b)^2)
    """
    spec_a = np.clip(_gd_spectrogram(audio_a), 0.0, 1.0).ravel()
    spec_b = np.clip(_gd_spectrogram(audio_b), 0.0, 1.0).ravel()
    b_dot_b = np.dot(spec_b, spec_b)
    if b_dot_b < 1e-10:
        return float(np.mean(spec_a ** 2))
    alpha = np.dot(spec_a, spec_b) / b_dot_b
    return float(np.mean((spec_a - alpha * spec_b) ** 2))


_JTFS_J = 6
_JTFS_Q = 1
_jtfs_scat = None
_jtfs_scat_len = None


def _get_jtfs_scattering(sample_rate: int, audio_len: int):
    """Lazy-build and cache the 1D scattering transform (matches GD: J=6, Q=1)."""
    global _jtfs_scat, _jtfs_scat_len
    if _jtfs_scat is None or _jtfs_scat_len != audio_len:
        from kymatio.numpy import Scattering1D
        _jtfs_scat = Scattering1D(J=_JTFS_J, shape=audio_len, Q=_JTFS_Q)
        _jtfs_scat_len = audio_len
    return _jtfs_scat


def jtfs_loss(audio_a: np.ndarray, audio_b: np.ndarray, sample_rate: int) -> float:
    """
    L2 distance between 1D scattering coefficients (J=6, Q=1).
    Matches the JTFS loss used by GD in agents/multidim/gradient_descent.py.
    """
    n = min(len(audio_a), len(audio_b))
    a = audio_a[:n].astype(np.float32)
    b = audio_b[:n].astype(np.float32)
    scat = _get_jtfs_scattering(sample_rate, n)
    sa = scat(a)
    sb = scat(b)
    return float(np.mean((sa - sb) ** 2))


def dtw_envelope_loss(audio_a: np.ndarray, audio_b: np.ndarray, sample_rate: int) -> float:
    """
    Hard DTW on GD onset envelopes.
    Matches SoftDTW(gamma=1) from softdtw_jax.py: squared L2 cost, no (n+m) normalization.
    """
    env_a = _gd_onset_envelope(audio_a)
    env_b = _gd_onset_envelope(audio_b)
    return _dtw_distance(env_a, env_b, squared=True, normalize=False)


# ------------------------------------------------------------------
# Registry for easy iteration
# ------------------------------------------------------------------

ALL_LOSSES = {
    "L2 Spectral": l2_spectral_loss,
    "Multi-Res Spectral": multi_resolution_spectral_loss,
    "SSIM Spectral": ssim_spectral_loss,
    "DTW Onset": dtw_onset_loss,
    "SIMSE_Spec": simse_spec_loss,
    "DTW_Envelope": dtw_envelope_loss,
    "JTFS": jtfs_loss,
}
