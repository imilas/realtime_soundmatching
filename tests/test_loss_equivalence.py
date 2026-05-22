"""
Tests that JAX differentiable losses and their numpy equivalents
agree on signal ranking (Spearman rank correlation).

We don't expect numerical equality — the STFT parameters and DTW variants
differ — but both should agree on which candidate is closer to the target.

Marked 'gd': requires JAX, dm_pix, DawDreamer.

    pytest tests/test_loss_equivalence.py -v -m gd
"""

import numpy as np
import pytest
from scipy.stats import spearmanr

pytestmark = pytest.mark.gd

SR = 44100
N_SAMPLES = SR  # 1 second
_NFFT = 512
_WIN_LEN = 600
_HOP_LEN = 100


@pytest.fixture(scope="module")
def audio_pairs():
    """
    Render bandpass_noise at 8 random parameter settings and pair each
    with a fixed target.  Returns (target_audio, list_of_candidate_audios).
    """
    from agent.params import FaustParams
    from experiments.multidim_runner import _bounds_from_params, _render_audio
    from synths.build import prepare

    build = prepare("bandpass_noise")
    params = FaustParams(str(build.json_path))
    bounds = _bounds_from_params(params)

    rng = np.random.default_rng(0)
    true_norm = rng.uniform(0, 1, size=bounds.d)
    true_real = bounds.denormalize(true_norm)
    target = _render_audio(str(build.dsp_path), params.vector_to_dict(true_real), N_SAMPLES, SR)

    candidates = []
    for _ in range(8):
        cand_norm = rng.uniform(0, 1, size=bounds.d)
        cand_real = bounds.denormalize(cand_norm)
        audio = _render_audio(str(build.dsp_path), params.vector_to_dict(cand_real), N_SAMPLES, SR)
        p_loss = float(np.linalg.norm(true_norm - cand_norm))
        candidates.append((audio, p_loss))

    return target, candidates


# ---------------------------------------------------------------------------
# SIMSE_Spec (JAX / dm_pix) vs SSIM Spectral (numpy)
# ---------------------------------------------------------------------------

def _jax_simse(target, pred):
    import jax.numpy as jnp
    import dm_pix
    from utils.jax_synth.loss_helpers import spec_func, clip_spec

    sf = spec_func(_NFFT, _WIN_LEN, _HOP_LEN)
    t = clip_spec(sf(jnp.array(target, dtype=jnp.float32))[0])[..., None]
    p = clip_spec(sf(jnp.array(pred,   dtype=jnp.float32))[0])[..., None]
    return float(dm_pix.simse(t, p))


def _numpy_ssim(target, pred):
    from utils.loss_functions import ssim_spectral_loss
    return ssim_spectral_loss(target, pred, sample_rate=SR)


def test_simse_zero_for_identical_signal():
    """SIMSE should be 0 when comparing a signal to itself."""
    rng = np.random.default_rng(1)
    audio = rng.standard_normal(N_SAMPLES).astype(np.float32)
    assert _jax_simse(audio, audio) < 1e-4


def test_ssim_zero_for_identical_signal():
    """SSIM loss should be 0 when comparing a signal to itself."""
    rng = np.random.default_rng(1)
    audio = rng.standard_normal(N_SAMPLES).astype(np.float32)
    assert _numpy_ssim(audio, audio) < 1e-4


def test_simse_ssim_rank_correlation(audio_pairs):
    """
    SIMSE and SSIM should rank candidates in the same order.
    Uses candidates at monotonically increasing P-Loss to reduce noise.
    """
    target, candidates = audio_pairs
    # Sort by P-Loss so we have a clean ground-truth ordering.
    candidates_sorted = sorted(candidates, key=lambda x: x[1])
    jax_scores   = [_jax_simse(target, cand) for cand, _ in candidates_sorted]
    numpy_scores = [_numpy_ssim(target, cand) for cand, _ in candidates_sorted]
    rho, pval = spearmanr(jax_scores, numpy_scores)
    print(f"\nSIMSE vs SSIM Spearman rho={rho:.3f} (p={pval:.3f})")
    print(f"  SIMSE scores:  {[round(x,4) for x in jax_scores]}")
    print(f"  SSIM  scores:  {[round(x,4) for x in numpy_scores]}")
    assert rho > 0.0, f"SIMSE vs SSIM have negative rank correlation: rho={rho:.3f}"


def test_simse_ssim_increase_with_distance(audio_pairs):
    """
    Both losses should be higher for a far candidate than a near one,
    at least on average.  Tests the monotonicity direction.
    """
    target, candidates = audio_pairs
    candidates_sorted = sorted(candidates, key=lambda x: x[1])
    near = candidates_sorted[:2]
    far  = candidates_sorted[-2:]

    near_simse = np.mean([_jax_simse(target, c) for c, _ in near])
    far_simse  = np.mean([_jax_simse(target, c) for c, _ in far])
    near_ssim  = np.mean([_numpy_ssim(target, c) for c, _ in near])
    far_ssim   = np.mean([_numpy_ssim(target, c) for c, _ in far])

    print(f"\nSIMSE: near={near_simse:.4f} far={far_simse:.4f}")
    print(f"SSIM:  near={near_ssim:.4f}  far={far_ssim:.4f}")
    assert far_simse > near_simse, f"SIMSE didn't increase with distance: near={near_simse:.4f} far={far_simse:.4f}"
    assert far_ssim  > near_ssim,  f"SSIM  didn't increase with distance: near={near_ssim:.4f}  far={far_ssim:.4f}"


# ---------------------------------------------------------------------------
# DTW_Envelope (JAX / SoftDTW) vs DTW Onset (numpy / hard DTW)
# ---------------------------------------------------------------------------

def _jax_dtw_envelope(target, pred):
    import jax.numpy as jnp
    from utils.jax_synth.loss_helpers import spec_func, gaussian_kernel1d, onset_1d
    from utils.jax_synth.softdtw_jax import SoftDTW

    sf = spec_func(_NFFT, _WIN_LEN, _HOP_LEN)
    kernel = jnp.array(gaussian_kernel1d(3, 0, 10))
    dtw = SoftDTW(gamma=1)
    t = jnp.array(target, dtype=jnp.float32)
    p = jnp.array(pred,   dtype=jnp.float32)
    return float(dtw(onset_1d(t, kernel, sf), onset_1d(p, kernel, sf)))


def _numpy_dtw_onset(target, pred):
    from utils.loss_functions import dtw_onset_loss
    return dtw_onset_loss(target, pred, sample_rate=SR)


def test_dtw_envelope_zero_for_identical_signal():
    """SoftDTW envelope loss should be near 0 for identical signals."""
    rng = np.random.default_rng(2)
    audio = rng.standard_normal(N_SAMPLES).astype(np.float32)
    val = _jax_dtw_envelope(audio, audio)
    assert val < 1e-3, f"DTW_Envelope on identical signals: {val}"


def test_dtw_onset_zero_for_identical_signal():
    """Hard DTW onset loss should be 0 for identical signals."""
    rng = np.random.default_rng(2)
    audio = rng.standard_normal(N_SAMPLES).astype(np.float32)
    val = _numpy_dtw_onset(audio, audio)
    assert val < 1e-6, f"DTW Onset on identical signals: {val}"


def test_dtw_envelope_onset_rank_correlation(audio_pairs):
    """
    SoftDTW envelope and hard DTW onset should rank candidates in the same order.
    """
    target, candidates = audio_pairs
    candidates_sorted = sorted(candidates, key=lambda x: x[1])
    jax_scores   = [_jax_dtw_envelope(target, cand) for cand, _ in candidates_sorted]
    numpy_scores = [_numpy_dtw_onset(target, cand)  for cand, _ in candidates_sorted]
    rho, pval = spearmanr(jax_scores, numpy_scores)
    print(f"\nDTW_Envelope vs DTW Onset Spearman rho={rho:.3f} (p={pval:.3f})")
    print(f"  JAX scores:   {[round(x,4) for x in jax_scores]}")
    print(f"  numpy scores: {[round(x,4) for x in numpy_scores]}")
    assert rho > 0.0, f"DTW_Envelope vs DTW Onset have negative rank correlation: rho={rho:.3f}"


def test_dtw_envelope_onset_increase_with_distance(audio_pairs):
    """
    Both DTW losses should be higher for far candidates than near ones.
    """
    target, candidates = audio_pairs
    candidates_sorted = sorted(candidates, key=lambda x: x[1])
    near = candidates_sorted[:2]
    far  = candidates_sorted[-2:]

    near_jax   = np.mean([_jax_dtw_envelope(target, c) for c, _ in near])
    far_jax    = np.mean([_jax_dtw_envelope(target, c) for c, _ in far])
    near_numpy = np.mean([_numpy_dtw_onset(target, c)  for c, _ in near])
    far_numpy  = np.mean([_numpy_dtw_onset(target, c)  for c, _ in far])

    print(f"\nDTW_Envelope: near={near_jax:.4f} far={far_jax:.4f}")
    print(f"DTW Onset:    near={near_numpy:.4f} far={far_numpy:.4f}")
    assert far_jax   > near_jax,   f"DTW_Envelope didn't increase with distance: near={near_jax:.4f} far={far_jax:.4f}"
    assert far_numpy > near_numpy, f"DTW Onset    didn't increase with distance: near={near_numpy:.4f} far={far_numpy:.4f}"
