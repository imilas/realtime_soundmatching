"""
Tests that numpy loss implementations match their JAX differentiable equivalents.

Marked 'gd': requires JAX, dm_pix, DawDreamer.

    pytest tests/test_loss_equivalence.py -v -m gd
"""

import numpy as np
import pytest

pytestmark = pytest.mark.gd

SR = 44100
N_SAMPLES = SR
_NFFT, _WIN_LEN, _HOP_LEN = 512, 600, 100


@pytest.fixture(scope="module")
def two_synth_signals():
    """Render bandpass_noise at two different parameter settings."""
    from agent.params import FaustParams
    from experiments.multidim_runner import _bounds_from_params, _render_audio
    from synths.build import prepare

    build = prepare("bandpass_noise")
    params = FaustParams(str(build.json_path))
    bounds = _bounds_from_params(params)
    rng = np.random.default_rng(42)

    def render(norm):
        real = bounds.denormalize(norm)
        return _render_audio(str(build.dsp_path), params.vector_to_dict(real), N_SAMPLES, SR)

    a = render(rng.uniform(0, 1, size=bounds.d))
    b = render(rng.uniform(0, 1, size=bounds.d))
    return a.astype(np.float32), b.astype(np.float32)


# ---------------------------------------------------------------------------
# SIMSE_Spec
# ---------------------------------------------------------------------------

def _jax_simse(a, b):
    import jax.numpy as jnp
    import dm_pix
    from utils.jax_synth.loss_helpers import spec_func, clip_spec
    sf = spec_func(_NFFT, _WIN_LEN, _HOP_LEN)
    ta = clip_spec(sf(jnp.array(a))[0])[..., None]
    tb = clip_spec(sf(jnp.array(b))[0])[..., None]
    return float(dm_pix.simse(ta, tb))


def _numpy_simse(a, b):
    from utils.loss_functions import simse_spec_loss
    return simse_spec_loss(a, b, SR)


def test_simse_identical_signal_is_zero(two_synth_signals):
    a, _ = two_synth_signals
    assert _numpy_simse(a, a) < 1e-6
    assert _jax_simse(a, a)   < 1e-6


def test_simse_numpy_matches_jax(two_synth_signals):
    a, b = two_synth_signals
    jax_val   = _jax_simse(a, b)
    numpy_val = _numpy_simse(a, b)
    print(f"\nSIMSE  JAX={jax_val:.6f}  numpy={numpy_val:.6f}  diff={abs(jax_val-numpy_val):.2e}")
    assert abs(jax_val - numpy_val) < 1e-4, (
        f"SIMSE mismatch: JAX={jax_val:.6f} numpy={numpy_val:.6f}"
    )


# ---------------------------------------------------------------------------
# DTW_Envelope
# ---------------------------------------------------------------------------

def _jax_dtw_envelope(a, b):
    import jax.numpy as jnp
    from utils.jax_synth.loss_helpers import spec_func, gaussian_kernel1d, onset_1d
    from utils.jax_synth.softdtw_jax import SoftDTW
    sf = spec_func(_NFFT, _WIN_LEN, _HOP_LEN)
    kernel = jnp.array(gaussian_kernel1d(3, 0, 10))
    dtw = SoftDTW(gamma=1)
    return float(dtw(
        onset_1d(jnp.array(a), kernel, sf),
        onset_1d(jnp.array(b), kernel, sf),
    ))


def _numpy_dtw_envelope(a, b):
    from utils.loss_functions import dtw_envelope_loss
    return dtw_envelope_loss(a, b, SR)


def test_dtw_envelope_identical_signal_is_zero(two_synth_signals):
    a, _ = two_synth_signals
    assert _numpy_dtw_envelope(a, a) < 1e-6
    # SoftDTW(gamma=1) on identical sequences is not exactly 0 but very small
    assert _jax_dtw_envelope(a, a) < 1e-3


def test_dtw_envelope_numpy_matches_jax(two_synth_signals):
    a, b = two_synth_signals
    jax_val   = _jax_dtw_envelope(a, b)
    numpy_val = _numpy_dtw_envelope(a, b)
    print(f"\nDTW_Envelope  JAX={jax_val:.6f}  numpy={numpy_val:.6f}  diff={abs(jax_val-numpy_val):.2e}")
    # Both use squared L2 cost; SoftDTW(gamma=1) soft-min vs hard-min differ by <1%.
    rel_diff = abs(jax_val - numpy_val) / (abs(jax_val) + 1e-10)
    assert rel_diff < 0.01, (
        f"DTW_Envelope relative mismatch: JAX={jax_val:.6f} numpy={numpy_val:.6f} rel={rel_diff:.3f}"
    )
