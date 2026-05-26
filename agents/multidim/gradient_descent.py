"""
Gradient descent sound-matching agent.

Requires .venv-gd (JAX, Flax, Optax, DawDreamer, kymatio, dm_pix).
All JAX imports are inside run_gd() so this module can be imported from .venv.
"""

from __future__ import annotations

import numpy as np

# Per-synth loss assignments from the IEEE 2025 paper.
SYNTH_LOSS_MAP: dict[str, str] = {
    "bandpass_noise": "SIMSE_Spec",
    "sine_saw": "JTFS",
    "am_noise": "DTW_Envelope",
    "sine_mod_saw": "JTFS",
    "sine_mod_sine": "JTFS",
}

_NFFT = 512
_WIN_LEN = 600
_HOP_LEN = 100
_SCAT_J = 6
_SCAT_Q = 1


def run_gd(
    dsp_code: str,
    target_audio: np.ndarray,
    init_real: dict[str, float],
    param_names: list[str],
    bounds_lowers: np.ndarray,
    bounds_uppers: np.ndarray,
    true_norm: np.ndarray,
    eval_budget: int,
    loss_name: str = "SIMSE_Spec",
    learning_rate: float = 0.045,
    seed: int = 0,
) -> tuple[list[float], list[float], list[dict[str, float]], dict[str, float]]:
    """
    Run RMSProp on the JAX-compiled Faust instrument.

    Each gradient step counts as one evaluation toward eval_budget.
    Real parameter values are extracted via DawDreamer intermediates after
    each step to compute P-Loss accurately.

    Parameters
    ----------
    dsp_code       : Faust DSP source text (with sliders, not baked constants)
    target_audio   : 1-D float array, the target waveform
    init_real      : {param_name: real_value} starting point for optimization
    param_names    : ordered list of param names (matches FaustParams ordering)
    bounds_lowers  : real lower bounds, shape (d,)
    bounds_uppers  : real upper bounds, shape (d,)
    true_norm      : true target in normalized [0,1]^d (for P-Loss)
    eval_budget    : number of gradient steps
    loss_name      : one of SIMSE_Spec | JTFS | DTW_Envelope
    learning_rate  : RMSProp step size
    seed           : JAX PRNG seed

    Returns
    -------
    history_audio_loss  : loss value at each step
    history_p_loss      : P-Loss at each step
    history_params      : real parameter values at each step
    best_params_real    : {param_name: value} of best point found
    """
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import train_state

    from utils.jax_synth.faust_to_jax import code_to_flax, SAMPLE_RATE
    from utils.jax_synth.loss_helpers import (
        spec_func, onset_1d, gaussian_kernel1d, naive_loss, clip_spec,
    )
    from utils.jax_synth.softdtw_jax import SoftDTW

    key = jax.random.PRNGKey(seed)
    # dsp_code must have init_real values baked in as hslider defaults so that
    # DawDreamer's Flax params (which represent offsets from the default, starting
    # at 0.0) begin exactly at the desired initial point.
    instrument, instrument_jit, noise, instrument_params = code_to_flax(dsp_code, key)

    # Build loss helpers.
    sf = spec_func(_NFFT, _WIN_LEN, _HOP_LEN)
    target_jax = jnp.array(target_audio, dtype=jnp.float32)

    if loss_name == "SIMSE_Spec":
        import dm_pix

        def _audio_loss(pred):
            # dm_pix.simse requires rank 3/4 (H,W,C or N,H,W,C); add channel dim.
            t_spec = clip_spec(sf(target_jax)[0])[..., None]
            p_spec = clip_spec(sf(pred)[0])[..., None]
            return dm_pix.simse(t_spec, p_spec)

    elif loss_name == "JTFS":
        from kymatio.jax import Scattering1D
        scat = Scattering1D(_SCAT_J, SAMPLE_RATE, _SCAT_Q)

        def _audio_loss(pred):
            return naive_loss(scat(target_jax), scat(pred))

    elif loss_name == "DTW_Envelope":
        dtw = SoftDTW(gamma=1)
        kernel = jnp.array(gaussian_kernel1d(3, 0, 10))

        def _audio_loss(pred):
            return dtw(
                onset_1d(target_jax, kernel, sf),
                onset_1d(pred, kernel, sf),
            )

    else:
        raise ValueError(f"Unknown loss_name: {loss_name!r}")

    def loss_fn(params):
        pred, _ = instrument_jit(params, noise, SAMPLE_RATE)
        return _audio_loss(pred), pred

    tx = optax.rmsprop(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=instrument.apply, params=instrument_params, tx=tx
    )

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    def _clip_grads(grads, clip_norm: float = 1.0):
        total = jnp.sqrt(
            sum(jnp.sum(p ** 2) for p in jax.tree_util.tree_leaves(grads))
        )
        scale = clip_norm / jnp.maximum(total, clip_norm)
        return jax.tree_util.tree_map(lambda g: g * scale, grads)

    @jax.jit
    def train_step(state):
        (loss, _pred), grads = grad_fn(state.params)
        grads = _clip_grads(grads, clip_norm=1.0)
        state = state.apply_gradients(grads=grads)
        return state, loss

    bounds_range = np.where(
        (bounds_uppers - bounds_lowers) == 0, 1.0, bounds_uppers - bounds_lowers
    )

    history_audio_loss: list[float] = []
    history_p_loss: list[float] = []
    history_params: list[dict[str, float]] = []
    best_p_loss = float("inf")
    best_params_real = dict(init_real)

    for _ in range(eval_budget):
        state, loss_val = train_step(state)
        loss_float = float(loss_val)

        # Forward pass to extract real param values from DawDreamer intermediates.
        # Keys: "dawdreamer/<param_name>" (no leading underscore).
        _audio, mod_vars = instrument_jit(state.params, noise, SAMPLE_RATE)
        cur_real = np.array([
            float(np.array(mod_vars["intermediates"].get(f"dawdreamer/{pname}", 0.0)).ravel()[0])
            for pname in param_names
        ])
        cur_norm = np.clip((cur_real - bounds_lowers) / bounds_range, 0.0, 1.0)
        p_loss = float(np.linalg.norm(true_norm - cur_norm))
        cur_params = {pname: float(cur_real[i]) for i, pname in enumerate(param_names)}

        history_audio_loss.append(loss_float)
        history_p_loss.append(p_loss)
        history_params.append(cur_params)

        if p_loss < best_p_loss:
            best_p_loss = p_loss
            best_params_real = cur_params

    return history_audio_loss, history_p_loss, history_params, best_params_real
