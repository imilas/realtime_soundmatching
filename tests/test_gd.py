"""
GD integration tests.

These tests require JAX, Flax, Optax, DawDreamer, kymatio, dm_pix — they
run against the real synth renderer, not a synthetic loss.  They are marked
with the 'gd' pytest marker so they can be run separately:

    pytest tests/test_gd.py -v -m gd
"""

import numpy as np
import pytest

pytestmark = pytest.mark.gd

N_SAMPLES = 44100
SR = 44100


@pytest.fixture(scope="module")
def bandpass_setup():
    """Compile bandpass_noise once for all tests in this module."""
    from agent.params import FaustParams
    from experiments.multidim_runner import _bounds_from_params, _render_audio
    from synths.build import prepare

    build = prepare("bandpass_noise")
    params = FaustParams(str(build.json_path))
    bounds = _bounds_from_params(params)
    return build, params, bounds


@pytest.fixture(scope="module")
def jax_warmed(bandpass_setup):
    """Run one short GD pass to warm up JAX JIT before the real tests."""
    from experiments.multidim_runner import _render_audio
    from synths.program import get_program

    build, params, bounds = bandpass_setup
    rng = np.random.default_rng(0)
    true_norm = rng.uniform(0, 1, size=bounds.d)
    true_real = bounds.denormalize(true_norm)
    true_params = params.vector_to_dict(true_real)
    target_audio = _render_audio(str(build.dsp_path), true_params, N_SAMPLES, SR)
    dsp_code = get_program("bandpass_noise").instantiate(true_params)
    from agents.multidim.gradient_descent import run_gd
    run_gd(
        dsp_code=dsp_code,
        target_audio=target_audio,
        init_real=true_params,
        param_names=params.names(),
        bounds_lowers=bounds.lowers,
        bounds_uppers=bounds.uppers,
        true_norm=true_norm,
        eval_budget=3,
        loss_name="SIMSE_Spec",
        seed=0,
    )
    return True


def _run_gd_direct(dsp_code, target_audio, init_real, params, bounds, true_norm,
                   budget=15, loss_name="SIMSE_Spec", lr=0.045):
    from agents.multidim.gradient_descent import run_gd
    return run_gd(
        dsp_code=dsp_code,
        target_audio=target_audio,
        init_real=init_real,
        param_names=params.names(),
        bounds_lowers=bounds.lowers,
        bounds_uppers=bounds.uppers,
        true_norm=true_norm,
        eval_budget=budget,
        loss_name=loss_name,
        learning_rate=lr,
        seed=0,
    )


def test_gd_audio_loss_decreases(bandpass_setup, jax_warmed):
    """Audio loss should be lower at the end than at the start."""
    from experiments.multidim_runner import _render_audio
    from synths.program import get_program

    build, params, bounds = bandpass_setup
    rng = np.random.default_rng(42)

    true_norm = rng.uniform(0, 1, size=bounds.d)
    true_params = params.vector_to_dict(bounds.denormalize(true_norm))
    init_norm = rng.uniform(0, 1, size=bounds.d)
    init_params = params.vector_to_dict(bounds.denormalize(init_norm))

    target_audio = _render_audio(str(build.dsp_path), true_params, N_SAMPLES, SR)
    dsp_code = get_program("bandpass_noise").instantiate(init_params)

    hist_audio, _, _ = _run_gd_direct(
        dsp_code, target_audio, init_params, params, bounds, true_norm, budget=20
    )

    assert hist_audio[-1] < hist_audio[0], (
        f"Audio loss did not decrease: start={hist_audio[0]:.4f} end={hist_audio[-1]:.4f}"
    )


def test_gd_p_loss_decreases(bandpass_setup, jax_warmed):
    """P-Loss should on average decrease over GD steps."""
    from experiments.multidim_runner import _render_audio
    from synths.program import get_program

    build, params, bounds = bandpass_setup
    rng = np.random.default_rng(7)

    true_norm = rng.uniform(0, 1, size=bounds.d)
    true_params = params.vector_to_dict(bounds.denormalize(true_norm))
    init_norm = rng.uniform(0, 1, size=bounds.d)
    init_params = params.vector_to_dict(bounds.denormalize(init_norm))

    target_audio = _render_audio(str(build.dsp_path), true_params, N_SAMPLES, SR)
    dsp_code = get_program("bandpass_noise").instantiate(init_params)

    _, hist_p, _ = _run_gd_direct(
        dsp_code, target_audio, init_params, params, bounds, true_norm, budget=20
    )

    assert hist_p[-1] < hist_p[0], (
        f"P-Loss did not decrease: start={hist_p[0]:.4f} end={hist_p[-1]:.4f}"
    )


def test_gd_perfect_init_stays_low(bandpass_setup, jax_warmed):
    """When init == true params, P-Loss should start near 0 and stay low."""
    from experiments.multidim_runner import _render_audio
    from synths.program import get_program

    build, params, bounds = bandpass_setup
    rng = np.random.default_rng(99)

    true_norm = rng.uniform(0, 1, size=bounds.d)
    true_params = params.vector_to_dict(bounds.denormalize(true_norm))
    target_audio = _render_audio(str(build.dsp_path), true_params, N_SAMPLES, SR)
    dsp_code = get_program("bandpass_noise").instantiate(true_params)

    _, hist_p, _ = _run_gd_direct(
        dsp_code, target_audio, true_params, params, bounds, true_norm, budget=10
    )

    assert hist_p[0] < 0.05, f"P-Loss at perfect init should be ~0, got {hist_p[0]:.4f}"
    assert min(hist_p) < 0.1, f"GD drifted away from perfect init: min P-Loss={min(hist_p):.4f}"


def test_gd_close_init_better_than_far_on_average(bandpass_setup, jax_warmed):
    """On average over multiple seeds, GD from a close init should reach lower P-Loss."""
    from experiments.multidim_runner import _render_audio
    from synths.program import get_program

    build, params, bounds = bandpass_setup
    n_seeds = 2
    budget = 15

    close_mins, far_mins = [], []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        true_norm = rng.uniform(0.2, 0.8, size=bounds.d)
        true_params = params.vector_to_dict(bounds.denormalize(true_norm))
        target_audio = _render_audio(str(build.dsp_path), true_params, N_SAMPLES, SR)

        close_norm = np.clip(true_norm + 0.08, 0, 1)
        close_params = params.vector_to_dict(bounds.denormalize(close_norm))
        far_norm = np.clip(1.0 - true_norm, 0, 1)
        far_params = params.vector_to_dict(bounds.denormalize(far_norm))

        _, hp_close, _ = _run_gd_direct(
            get_program("bandpass_noise").instantiate(close_params),
            target_audio, close_params, params, bounds, true_norm, budget=budget
        )
        _, hp_far, _ = _run_gd_direct(
            get_program("bandpass_noise").instantiate(far_params),
            target_audio, far_params, params, bounds, true_norm, budget=budget
        )
        close_mins.append(min(hp_close))
        far_mins.append(min(hp_far))

    mean_close = np.mean(close_mins)
    mean_far = np.mean(far_mins)
    assert mean_close < mean_far, (
        f"Expected close init to beat far init on average: "
        f"close={mean_close:.4f} far={mean_far:.4f}"
    )


def test_gd_loss_name_passed_through():
    """run_trial_gd should use the loss_name passed in, not the hardcoded default."""
    from experiments.multidim_runner import run_trial_gd

    result = run_trial_gd(
        program_name="bandpass_noise",
        seed=0,
        eval_budget=3,
        loss_name="SIMSE_Spec",
    )
    assert result.loss_name == "SIMSE_Spec"
