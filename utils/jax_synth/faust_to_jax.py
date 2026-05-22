"""
Compile a Faust DSP program to a differentiable JAX/Flax module via DawDreamer.

Adapted from `audio_nexting/helper_funcs/faust_to_jax.py`. Stripped of
IPython/matplotlib/marimo dependencies so it works as a plain library.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from dawdreamer.faust import FaustContext
import dawdreamer.faust.box as fbox


SAMPLE_RATE = 44100
jax.config.update("jax_platform_name", "cpu")


def faust2jax(faust_code: str, module_name: str = "MyDsp"):
    """Compile Faust source string to a Flax module class."""
    with FaustContext():
        box = fbox.boxFromDSP(faust_code)
        jax_code = fbox.boxToSource(box, "jax", module_name, ["-a", "jax/minimal.py"])

    custom_globals: dict = {}
    exec(jax_code, custom_globals)  # required: DawDreamer generates Python source
    return custom_globals[module_name]


def code_to_flax(faust_code: str, key: jax.Array, length_seconds: float = 1.0):
    """
    Compile and initialize a Flax instrument from Faust code.

    Returns
    -------
    instrument : Flax module instance
    instrument_jit : JIT-compiled apply function
    noise : zero-or-random input tensor (some synths take inputs, most don't)
    instrument_params : initial FrozenDict of parameters
    """
    DSP = faust2jax(faust_code)
    instrument = DSP(SAMPLE_RATE)
    n_samples = int(SAMPLE_RATE * length_seconds)
    noise = jax.random.uniform(
        key,
        [instrument.getNumInputs(), n_samples],
        minval=-1.0,
        maxval=1.0,
    )
    instrument_params = instrument.init(key, noise, SAMPLE_RATE)
    instrument_jit = jax.jit(
        partial(instrument.apply, mutable="intermediates"),
        static_argnums=[2],
    )
    return instrument, instrument_jit, noise, instrument_params


def list_dawdreamer_params(params) -> list[str]:
    """Return the slider names registered by DawDreamer (without the prefix)."""
    keys = list(params["params"].keys())
    # DawDreamer keys look like '_dawdreamer/lp_cut'
    return [k.split("/", 1)[1] if "/" in k else k for k in keys]
