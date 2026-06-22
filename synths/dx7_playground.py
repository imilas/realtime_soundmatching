"""
DX7-style FM synth playground (marimo).

Interactive notebook to audition a DX7-inspired, fully-differentiable FM
synthesizer before deciding whether to add it as an official benchmark synth.

Run:
    conda activate soundmatch
    source experiment_scripts/env_capped.sh
    marimo edit synths/dx7_playground.py

Background
----------
The full upstream Faust DX7 (`dx.algorithm(1..32)`) renders with the normal
compiler but SEGFAULTS in DawDreamer's Faust->JAX backend, so it can't be
differentiated in this pipeline. This is instead a reduced 4-operator FM synth
(the same strategy as DDX7, ISMIR 2022) built from naive phasor oscillators.
FM is just nested sines (no comparison / floor / sample-and-hold), so every
parameter is differentiable a.e. -- the gradient-check cell at the bottom proves it.
It renders through the exact same `faust2jax` path the benchmark uses.

Note: operator feedback (the DX7's op-6 self-loop) is deliberately omitted -- it
renders fine but its gradient goes NaN through the long `~` recursion, which would
violate the "all params differentiable" requirement.
"""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import functools
    import io
    import sys
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import numpy as np
    import soundfile as sf
    import matplotlib.pyplot as plt
    import marimo as mo

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.jax_synth.faust_to_jax import faust2jax, SAMPLE_RATE

    return SAMPLE_RATE, faust2jax, functools, io, jax, jnp, mo, np, plt, sf


@app.cell
def _(mo):
    mo.md("""
    # 🎛️ DX7-style FM synth playground

    A reduced **4-operator FM** synth (DDX7-style), fully differentiable,
    rendered through this project's `faust2jax` path. Pick an *algorithm*
    (operator routing), move the sliders, listen, and check the spectrum.
    The bottom cell proves every control carries a gradient.
    """)
    return


@app.cell
def _():
    # Shared param ranges (must match the hslider declarations below). The
    # DawDreamer JAX backend stores each param normalized to [-1, 1], so we
    # convert raw slider values with: norm = 2*(v-lo)/(hi-lo) - 1.
    RANGES = dict(
        f0=(50.0, 1000.0),
        r1=(0.25, 8.0), r2=(0.25, 8.0), r3=(0.25, 8.0), r4=(0.25, 8.0),
        l1=(0.0, 8.0), l2=(0.0, 8.0), l3=(0.0, 8.0), l4=(0.0, 8.0),
    )

    HEADER = '''
    import("stdfaust.lib");
    f0 = hslider("f0",220,50,1000,0.1);
    r1 = hslider("r1",1,0.25,8,0.01);
    r2 = hslider("r2",2,0.25,8,0.01);
    r3 = hslider("r3",3,0.25,8,0.01);
    r4 = hslider("r4",4,0.25,8,0.01);
    l1 = hslider("l1",1,0,8,0.01);
    l2 = hslider("l2",2,0,8,0.01);
    l3 = hslider("l3",2,0,8,0.01);
    l4 = hslider("l4",1,0,8,0.01);
    phasor(f) = +(f/ma.SR) ~ ma.frac;
    op(f,pm)  = phasor(f)*2*ma.PI + pm : sin;   // phase-modulated sine operator
    '''

    # Operator routings. l_k is a carrier amplitude or a modulation index
    # depending on where operator k sits in the routing.
    ALGOS = {
        "1: serial stack (4→3→2→1)": HEADER + '''
            m4 = op(f0*r4, 0)*l4;
            m3 = op(f0*r3, m4)*l3;
            m2 = op(f0*r2, m3)*l2;
            process = op(f0*r1, m2)*l1 * 0.3;
        ''',
        "2: two pairs (2→1)+(4→3)": HEADER + '''
            vA = op(f0*r1, op(f0*r2, 0)*l2)*l1;
            vB = op(f0*r3, op(f0*r4, 0)*l4)*l3;
            process = (vA + vB) * 0.3;
        ''',
        "3: three mods → one carrier": HEADER + '''
            mods = op(f0*r2,0)*l2 + op(f0*r3,0)*l3 + op(f0*r4,0)*l4;
            process = op(f0*r1, mods)*l1 * 0.3;
        ''',
    }
    return ALGOS, RANGES


@app.cell
def _(ALGOS, RANGES, SAMPLE_RATE, faust2jax, functools, jax):
    # Compile each algorithm once (slow); cache by name.
    @functools.lru_cache(maxsize=None)
    def compile_algo(algo_name):
        DSP = faust2jax(ALGOS[algo_name])
        inst = DSP(SAMPLE_RATE)
        key = jax.random.PRNGKey(0)
        noise = jax.random.uniform(
            key, [inst.getNumInputs(), SAMPLE_RATE], minval=-1, maxval=1
        )
        params = inst.init(key, noise, SAMPLE_RATE)
        apply = jax.jit(functools.partial(inst.apply), static_argnums=[2])
        return inst, apply, params

    def set_params(template, values):
        """Write raw slider values into the param dict, normalized to [-1,1]."""
        import flax
        out = {}
        for k, v in template["params"].items():
            name = k.split("/", 1)[1] if "/" in k else k
            if name in values:
                lo, hi = RANGES[name]
                out[k] = 2.0 * (float(values[name]) - lo) / (hi - lo) - 1.0
            else:
                out[k] = v
        return flax.core.freeze({"params": out})

    return compile_algo, set_params


@app.cell
def _(ALGOS, mo):
    algo = mo.ui.dropdown(options=list(ALGOS), value=list(ALGOS)[0], label="Algorithm")
    dur = mo.ui.slider(0.3, 3.0, value=1.0, step=0.1, label="duration (s)")
    f0 = mo.ui.slider(50, 1000, value=220, step=1, label="f0 (Hz)")
    r1 = mo.ui.slider(0.25, 8, value=1.0, step=0.25, label="ratio 1 (carrier)")
    r2 = mo.ui.slider(0.25, 8, value=2.0, step=0.25, label="ratio 2")
    r3 = mo.ui.slider(0.25, 8, value=3.0, step=0.25, label="ratio 3")
    r4 = mo.ui.slider(0.25, 8, value=4.0, step=0.25, label="ratio 4")
    l1 = mo.ui.slider(0, 8, value=1.0, step=0.05, label="level 1 (carrier amp)")
    l2 = mo.ui.slider(0, 8, value=2.0, step=0.05, label="level 2 (index)")
    l3 = mo.ui.slider(0, 8, value=2.0, step=0.05, label="level 3 (index)")
    l4 = mo.ui.slider(0, 8, value=1.0, step=0.05, label="level 4 (index)")

    controls = mo.vstack([
        algo,
        mo.hstack([f0, dur], justify="start"),
        mo.hstack([r1, r2, r3, r4], justify="start"),
        mo.hstack([l1, l2, l3, l4], justify="start"),
    ])
    controls
    return algo, dur, f0, l1, l2, l3, l4, r1, r2, r3, r4


@app.cell
def _(
    SAMPLE_RATE,
    algo,
    compile_algo,
    dur,
    f0,
    io,
    jnp,
    l1,
    l2,
    l3,
    l4,
    mo,
    np,
    plt,
    r1,
    r2,
    r3,
    r4,
    set_params,
    sf,
):
    vals = dict(
        f0=f0.value, r1=r1.value, r2=r2.value, r3=r3.value, r4=r4.value,
        l1=l1.value, l2=l2.value, l3=l3.value, l4=l4.value,
    )
    _inst, _apply, _template = compile_algo(algo.value)
    _n = int(SAMPLE_RATE * dur.value)
    _sig = np.asarray(
        _apply(set_params(_template, vals), jnp.zeros((_inst.getNumInputs(), _n)), SAMPLE_RATE)
    ).ravel()

    _peak = float(np.abs(_sig).max()) or 1.0
    _audio = (0.9 * _sig / _peak).astype(np.float32)
    _buf = io.BytesIO(); sf.write(_buf, _audio, SAMPLE_RATE, format="WAV"); _buf.seek(0)

    _fig, (_axw, _axs) = plt.subplots(1, 2, figsize=(11, 3))
    _k = min(len(_sig), int(0.02 * SAMPLE_RATE))
    _axw.plot(np.arange(_k) / SAMPLE_RATE * 1000, _sig[:_k])
    _axw.set_xlabel("ms"); _axw.set_title("waveform (20 ms)")
    _spec = np.abs(np.fft.rfft(_sig * np.hanning(len(_sig))))
    _freqs = np.fft.rfftfreq(len(_sig), 1 / SAMPLE_RATE)
    _m = _freqs < 8000
    _axs.plot(_freqs[_m], 20 * np.log10(_spec[_m] + 1e-9))
    _axs.set_xlabel("Hz"); _axs.set_title("magnitude spectrum (dB)")
    _fig.tight_layout()

    mo.vstack([
        mo.md(f"**peak (pre-norm):** {_peak:.3f}  •  **rms:** {np.sqrt((_sig**2).mean()):.3f}"),
        mo.audio(_buf),
        _fig,
    ])
    return (vals,)


@app.cell
def _(mo):
    mo.md("""
    ---
    ### ✅ Differentiability check
    Per-parameter gradient magnitude of `sum(audio²)` at the current settings.
    All nonzero ⇒ every control is GD-trainable. (A control reads 0 only when
    its operator is silenced by a `level = 0`.)
    """)
    return


@app.cell
def _(SAMPLE_RATE, algo, compile_algo, jax, jnp, mo, set_params, vals):
    _inst2, _apply2, _template2 = compile_algo(algo.value)
    _n2 = SAMPLE_RATE // 2
    _noise2 = jnp.zeros((_inst2.getNumInputs(), _n2))

    def _loss(p):
        return jnp.sum(_inst2.apply(p, _noise2, SAMPLE_RATE) ** 2)

    _g = jax.grad(_loss)(set_params(_template2, vals))["params"]
    _rows = []
    for _k in sorted(_g):
        _name = _k.split("/", 1)[1] if "/" in _k else _k
        _rows.append(f"| `{_name}` | {float(jnp.abs(jnp.ravel(_g[_k])).sum()):.3e} |")
    mo.md("| param | \\|grad\\| |\n|---|---|\n" + "\n".join(_rows))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
