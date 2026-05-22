"""
GD integration tests.

These tests require JAX, Flax, Optax, DawDreamer, kymatio, dm_pix — they
run against the real synth renderer, not a synthetic loss.  They are slow
(~30-60s each) and marked with the 'gd' pytest marker so they can be run
separately:

    pytest tests/test_gd.py -v -m gd
"""

import numpy as np
import pytest

pytestmark = pytest.mark.gd


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


def _run_gd_direct(dsp_code, target_audio, init_real, params, bounds, true_norm,
                   budget=50, loss_name="SIMSE_Spec", lr=0.045):
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


def test_gd_audio_loss_decreases(bandpass_setup):
    """Audio loss should be lower at the end than at the start."""
    from experiments.multidim_runner import _render_audio
    from synths.program import get_program

    build, params, bounds = bandpass_setup
    rng = np.random.default_rng(42)

    true_norm = rng.uniform(0, 1, size=bounds.d)
    true_real = bounds.denormalize(true_norm)
    true_params = params.vector_to_dict(true_real)

    init_norm = rng.uniform(0, 1, size=bounds.d)
    init_real = bounds.denormalize(init_norm)
    init_params = params.vector_to_dict(init_real)

    n_samples = 44100
    target_audio = _render_audio(str(build.dsp_path), true_params, n_samples, 44100)
    dsp_code = get_program("bandpass_noise").instantiate(init_params)

    hist_audio, hist_p, _ = _run_gd_direct(
        dsp_code, target_audio, init_params, params, bounds, true_norm, budget=80
    )

    first_quarter = np.mean(hist_audio[:20])
    last_quarter = np.mean(hist_audio[-20:])
    assert last_quarter < first_quarter, (
        f"Audio loss did not decrease: start={first_quarter:.4f} end={last_quarter:.4f}"
    )


def test_gd_p_loss_decreases(bandpass_setup):
    """P-Loss should on average decrease over GD steps."""
    from experiments.multidim_runner import _render_audio
    from synths.program import get_program

    build, params, bounds = bandpass_setup
    rng = np.random.default_rng(7)

    true_norm = rng.uniform(0, 1, size=bounds.d)
    true_real = bounds.denormalize(true_norm)
    true_params = params.vector_to_dict(true_real)

    init_norm = rng.uniform(0, 1, size=bounds.d)
    init_real = bounds.denormalize(init_norm)
    init_params = params.vector_to_dict(init_real)

    n_samples = 44100
    target_audio = _render_audio(str(build.dsp_path), true_params, n_samples, 44100)
    dsp_code = get_program("bandpass_noise").instantiate(init_params)

    _, hist_p, _ = _run_gd_direct(
        dsp_code, target_audio, init_params, params, bounds, true_norm, budget=100
    )

    first_half = np.mean(hist_p[:50])
    second_half = np.mean(hist_p[50:])
    assert second_half < first_half, (
        f"P-Loss did not decrease: first_half={first_half:.4f} second_half={second_half:.4f}"
    )


def test_gd_perfect_init_stays_low(bandpass_setup):
    """When init == true params, P-Loss should start near 0 and stay low."""
    from experiments.multidim_runner import _render_audio
    from synths.program import get_program

    build, params, bounds = bandpass_setup
    rng = np.random.default_rng(99)

    true_norm = rng.uniform(0, 1, size=bounds.d)
    true_real = bounds.denormalize(true_norm)
    true_params = params.vector_to_dict(true_real)

    n_samples = 44100
    target_audio = _render_audio(str(build.dsp_path), true_params, n_samples, 44100)
    # Init exactly at the true params — P-Loss should stay near 0.
    dsp_code = get_program("bandpass_noise").instantiate(true_params)

    _, hist_p, best_params = _run_gd_direct(
        dsp_code, target_audio, true_params, params, bounds, true_norm, budget=30
    )

    assert hist_p[0] < 0.05, f"P-Loss at perfect init should be ~0, got {hist_p[0]:.4f}"
    assert min(hist_p) < 0.1, f"GD drifted away from perfect init: min P-Loss={min(hist_p):.4f}"


def test_gd_close_init_better_than_far_on_average(bandpass_setup):
    """
    On average over multiple seeds, GD from a close init should reach lower P-Loss
    than GD from a far init.  A single-seed comparison is unreliable because the
    loss landscape can be non-convex.
    """
    from experiments.multidim_runner import _render_audio
    from synths.program import get_program

    build, params, bounds = bandpass_setup
    n_samples = 44100
    n_seeds = 5
    budget = 80

    close_mins, far_mins = [], []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        true_norm = rng.uniform(0.2, 0.8, size=bounds.d)  # keep target away from edges
        true_real = bounds.denormalize(true_norm)
        true_params = params.vector_to_dict(true_real)
        target_audio = _render_audio(str(build.dsp_path), true_params, n_samples, 44100)

        # Close init: 0.08 away from true in a fixed direction.
        close_norm = np.clip(true_norm + 0.08, 0, 1)
        close_real = bounds.denormalize(close_norm)
        close_params = params.vector_to_dict(close_real)

        # Far init: opposite side of the space from true.
        far_norm = np.clip(1.0 - true_norm, 0, 1)
        far_real = bounds.denormalize(far_norm)
        far_params = params.vector_to_dict(far_real)

        dsp_close = get_program("bandpass_noise").instantiate(close_params)
        dsp_far = get_program("bandpass_noise").instantiate(far_params)

        _, hp_close, _ = _run_gd_direct(
            dsp_close, target_audio, close_params, params, bounds, true_norm, budget=budget
        )
        _, hp_far, _ = _run_gd_direct(
            dsp_far, target_audio, far_params, params, bounds, true_norm, budget=budget
        )
        close_mins.append(min(hp_close))
        far_mins.append(min(hp_far))

    mean_close = np.mean(close_mins)
    mean_far = np.mean(far_mins)
    assert mean_close < mean_far, (
        f"Expected close init to beat far init on average: "
        f"close={mean_close:.4f} far={mean_far:.4f}\n"
        f"per-seed close={[round(x,4) for x in close_mins]} "
        f"far={[round(x,4) for x in far_mins]}"
    )


def test_gd_loss_name_passed_through():
    """run_trial_gd should use the loss_name passed in, not the hardcoded default."""
    from experiments.multidim_runner import run_trial_gd

    # Run with explicit loss_name — should not raise.
    result = run_trial_gd(
        program_name="bandpass_noise",
        seed=0,
        eval_budget=10,
        loss_name="SIMSE_Spec",
    )
    assert result.loss_name == "SIMSE_Spec"

    result2 = run_trial_gd(
        program_name="bandpass_noise",
        seed=0,
        eval_budget=10,
        loss_name="JTFS",
    )
    assert result2.loss_name == "JTFS"
