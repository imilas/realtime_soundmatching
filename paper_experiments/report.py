import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import pickle
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from utils.notebooks.trajectory_funcs import (
        load_all_results, build_trial_context, extract_trajectory,
        compute_surface, plot_full_trajectory,
    )

    RES  = Path(__file__).parent / "results"
    FIG  = RES / "figures"
    METHODS = ["GD", "RandomSearch", "CMA-ES", "BO"]
    SYNTHS  = ["bandpass_noise", "am_noise", "add_sinesaw"]
    COLORS  = {"GD": "black", "RandomSearch": "#ff7f0e", "CMA-ES": "#2ca02c", "BO": "#d62728", "Learned": "#e377c2"}
    return (
        FIG,
        METHODS,
        RES,
        build_trial_context,
        compute_surface,
        extract_trajectory,
        load_all_results,
        mo,
        np,
        pd,
        pickle,
        plot_full_trajectory,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Sound-matching benchmark report

    **Comparing gradient descent with black-box optimisation methods for
    synthesizer parameter estimation.**

    ---

    ## Overview

    Given a target audio recording produced by a known synthesizer, the goal is
    to recover the synthesizer parameters that reproduce it as closely as
    possible.  We benchmark four methods:

    | Method | Type | Requires differentiable synth? |
    |---|---|---|
    | **GD** — RMSProp on a JAX-compiled synth | Gradient-based | Yes |
    | **CMA-ES** — Covariance Matrix Adaptation ES | Black-box | No |
    | **BO** — Bayesian Optimisation (GP + EI) | Black-box | No |
    | **RandomSearch** — Uniform sampling | Black-box baseline | No |

    We also report a **learned amortized inverse model** (zero-shot; trained
    offline) as a preview of the direction beyond search.

    All methods run for **200 evaluations per trial**, **200 matched trials per
    cell** (same random targets across methods, seeded by trial index).
    Parameter-loss (P-loss) = Euclidean distance in normalised [0,1]² space
    (0 = exact match, √2 ≈ 1.41 = worst possible).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 1. Experimental setup

    ### 1.1 Synthesizers

    Three Faust synthesizers chosen as **loss-landscape archetypes**:

    | Synth | Parameters | Loss | Landscape |
    |---|---|---|---|
    | `bandpass_noise` | `hp_cut` [30–5 000 Hz], `lp_cut` [60–6 000 Hz] | SIMSE_Spec | Smooth, identifiable — tests local exploitation |
    | `am_noise` | `amp` [0.1–1], `carrier` [10–1 000 Hz] | DTW_Envelope | Moderate — tests explore/exploit balance |
    | `add_sinesaw` | `saw_freq` [20–1 000 Hz], `sine_freq` [20–1 000 Hz] | SIMSE_Spec | **Flat / non-identifiable** — tests robustness to a deceptive loss |

    All renderings are deterministic (verified: identical audio for identical
    parameters on all three synths).

    ### 1.2 Losses

    Two perceptual losses are used, each with a NumPy implementation (for the
    black-box methods) and a JAX-differentiable implementation (for GD).

    **Loss-equivalence verification** (`tests/test_loss_equivalence.py`):

    | Loss | NumPy vs JAX absolute diff | Verdict |
    |---|---|---|
    | SIMSE_Spec | 1.18 × 10⁻¹⁰ | Effectively identical |
    | DTW_Envelope | 0.11 % relative | Near-identical (soft vs hard DTW) |

    The comparison between GD and gradient-free methods is therefore valid —
    they optimise the same objective.

    ### 1.3 Two accuracy metrics

    We distinguish two P-loss quantities that are often conflated:

    - **Visited** — best parameters ever sampled *(requires oracle knowledge of
      the true params; optimistic upper bound)*.
    - **Returned** — parameters the method would actually deploy = argmin of
      audio loss *(what matters in practice)*.

    Their difference is the **deception gap**: how badly the audio loss
    misleads a method into returning wrong parameters.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 2. Method implementations

    All methods operate in **normalised parameter space** [0, 1]^d.  The runner
    denormalises to real parameter values before each audio render.

    ---

    ### 2.1 Gradient Descent (GD)

    **Pipeline:** Faust DSP source → JAX/Flax compiled instrument via DawDreamer
    → differentiable audio render → perceptual loss → RMSProp update.

    **Optimiser:** `optax.rmsprop`, learning rate **0.045**, gradient clipping
    at global norm **1.0** (applied before the update step).

    **Initialisation:** the DSP source is rewritten with the initial parameter
    values baked in as `hslider` defaults so that Flax's internal parameters
    (which represent offsets from the default, initialised to 0) begin exactly
    at the desired starting point.

    **Spectrogram config (SIMSE_Spec path):**
    FFT size 512, window length 600, hop 100.

    **Scattering config (JTFS path, unused in this benchmark):**
    J = 6, Q = 1.

    **Real-parameter extraction:** after each gradient step, a second forward
    pass extracts the synthesizer's intermediate parameter values from
    DawDreamer's `intermediates` dict (`"dawdreamer/<param_name>"`).  This
    pass is the main source of GD's high per-step wall-clock cost.

    **Key constraint:** the synthesizer must be compiled to a JAX-differentiable
    Flax model.  This rules out VSTs, hardware synthesizers, and any non-Faust DSP.

    ---

    ### 2.2 CMA-ES

    **Library:** `cma` (Python, Hansen et al.).

    **Initial mean:** centre of the box, [0.5, 0.5] in normalised space.

    **Initial step-size σ₀ = 0.3** — covers most of the unit box without
    concentrating samples at corners after clipping.

    **Population size:** CMA-ES default, `4 + floor(3 · ln d)` = **6** for d = 2.

    **Stopping criteria:** `tolfun = tolx = 1e-12` — effectively disabled so
    the evaluation budget is the only termination condition.

    **Ask/tell interface:** the `cma` library is generation-based.  The wrapper
    (`agents/multidim/cma_es.py`) buffers one generation's proposals in a deque;
    `propose()` pops from the front, `observe()` accumulates losses and calls
    `es.tell()` when one full generation has been evaluated.

    **Bounds:** hard-clipped to [0, 1]^d by the `cma` `bounds` option.

    ---

    ### 2.3 Bayesian Optimisation (BO)

    **Library:** `scikit-optimize` (`skopt.Optimizer`).

    **Surrogate:** Gaussian Process with Matérn 5/2 kernel, **Gaussian
    noise estimated from the data** (skopt default).

    **Acquisition function:** Expected Improvement (EI), maximised by random
    sampling inside skopt.

    **Initial random phase:** **10 uniformly-sampled points** before the GP
    is first fitted.

    **Ask/tell:** one observation per step — `propose()` calls `opt.ask()`,
    `observe()` calls `opt.tell()`.

    **Known limitation:** the default GP estimates observation noise.  Since
    audio rendering is near-deterministic, this inflates posterior uncertainty
    and keeps EI in a global exploration mode throughout, producing large
    (corner-to-corner) step sizes.  A version with `noise ≈ 1e-10` was
    validated and shows improvement (halves am_noise returned P-loss), but
    is not used in the main benchmark to keep the baseline representative.

    ---

    ### 2.4 Random Search (RS)

    Each proposal is drawn **independently and uniformly** from [0, 1]^d,
    regardless of all previous observations.  This is the standard
    random-search floor — it provides an unbiased coverage of parameter space
    and is the correct lower-bound baseline.

    *Note:* an earlier implementation used a Gaussian random walk (σ = 0.1)
    that reset to the last observed point.  That walk ignores the objective,
    limits exploration radius, and produced best-so-far curves independent of
    the synth — it has been replaced with uniform sampling.

    ---

    ### 2.5 Learned Amortized Inverse Model

    **Concept:** instead of searching at test time, train a model offline to
    invert the synthesizer: given audio features, predict the parameters.

    **Training data:** 15 000 `(params → audio)` pairs per synth, with
    parameters sampled uniformly from [0, 1]^d.  Supervision is on the
    normalised parameters directly (not on audio loss), so the model learns
    the true prior without the audio loss acting as an intermediary.

    **Feature extraction** (`agents/learned/data_gen.py`):
    scipy STFT (Hann window, FFT 2048, hop 512) → log-magnitude spectrogram →
    block-mean pooled to **64 frequency × 8 time bins** (512-d) → per-clip
    z-scored.  The 8 time bins preserve the temporal envelope structure needed
    for `am_noise`.

    **Model** (`sklearn.neural_network.MLPRegressor`):
    two hidden layers **[256, 128]**, ReLU activations, Adam optimiser,
    up to 300 epochs with early stopping, output clipped to [0, 1].

    **Evaluation:** zero-shot on the exact 200 benchmark targets (same seeds as
    the optimisers) — 0 audio evaluations at deploy time.

    **Deployment cost:** one forward pass (~1 ms) after a one-time training
    cost of ~15 min (15 k renders + MLP training).
    """)
    return


@app.cell
def _(FIG, mo):
    mo.vstack([
        mo.md(r"""
    ---
    ## 3. Final accuracy

    Best P-loss (visited) distribution across 200 trials per method.
    Log scale — lower is better.  Boxes show IQR; whiskers 5th–95th percentile.
        """),
        mo.image(str(FIG / "01_boxplots_bestploss.png")),
        mo.md(r"""
    **Key observations:**

    - On `bandpass_noise` **CMA-ES** (median 0.001) and **BO** (0.041) dominate;
      GD (0.025) is mid-table but has a heavy failure tail (mean 0.22).
    - On `am_noise` GD barely moves from its initialisation (median 0.376 visited
      after 200 gradient steps) — the DTW gradient signal is too weak at these
      step sizes.
    - On `add_sinesaw` *all methods are weak* — this is intentional: the loss
      surface is nearly flat (see §6 Non-identifiability).
        """),
    ])
    return


@app.cell
def _(FIG, mo):
    mo.vstack([
        mo.md(r"""
    ---
    ## 4. Deception gap — returned vs visited

    Grey bars = best parameter ever **visited** (oracle).
    Red bars  = parameter the method would **return** (argmin audio loss).
    The gap between them measures how badly the audio loss fools a method.
        """),
        mo.image(str(FIG / "02_returned_vs_visited.png")),
        mo.md(r"""
    | Synth | CMA-ES gap | BO gap | GD gap | RS gap |
    |---|---|---|---|---|
    | bandpass_noise | **0.000** | 0.017 | 0.062 | 0.026 |
    | am_noise | **0.005** | 0.007 | 0.112 | 0.015 |
    | add_sinesaw | **0.298** | 0.176 | 0.153 | 0.118 |

    **Mechanism.**
    On identifiable synths the gap is near zero — the loss guides search
    correctly, and convergent methods (CMA-ES) dominate.
    On `add_sinesaw` every method *visits* near-perfect parameters (visited
    median 0.02–0.04) but *returns* wrong ones.  CMA-ES has the **largest
    gap (0.298)** because it commits to a single basin; once it converges to
    the wrong local minimum the loss cannot redirect it.  BO hedges globally
    and has a smaller gap (0.176).

    This is the central finding: **the strength of convergent methods (fast
    local search) becomes their weakness on deceptive losses.**
        """),
    ])
    return


@app.cell
def _(FIG, mo):
    mo.vstack([
        mo.md(r"""
    ---
    ## 5. Sample efficiency

    Median best-so-far P-loss (visited) versus evaluation count.
    IQR band shown.  Log y-axis.
        """),
        mo.image(str(FIG / "03_learning_curves.png")),
        mo.md(r"""
    **Snapshot table — median best-so-far at fixed budgets:**

    | Synth | Method | @25 | @50 | @100 | @200 |
    |---|---|---|---|---|---|
    | bandpass | GD | 0.128 | 0.089 | 0.045 | 0.025 |
    | bandpass | RS | 0.095 | 0.066 | 0.047 | 0.034 |
    | bandpass | **CMA-ES** | **0.083** | **0.041** | **0.012** | **0.000** |
    | bandpass | BO | 0.076 | 0.050 | 0.035 | 0.025 |
    | am_noise | GD | 0.386 | 0.384 | 0.377 | 0.376 |
    | am_noise | RS | 0.095 | 0.066 | 0.047 | 0.034 |
    | am_noise | **CMA-ES** | **0.082** | **0.046** | **0.017** | **0.004** |
    | am_noise | **BO** | 0.084 | 0.054 | 0.021 | **0.007** |

    GD's curve on `am_noise` is essentially flat — 200 gradient steps produce
    almost no improvement from the initialisation.
        """),
    ])
    return


@app.cell
def _(FIG, mo):
    mo.vstack([
        mo.md(r"""
    ---
    ## 6. Wall-clock efficiency (controlled conditions)

    Per-eval cost measured under **controlled, interleaved conditions**
    (all methods timed back-to-back in the same run) using a two-budget slope
    to separate per-step cost from fixed setup/JIT overhead.

    Bar height = ms per evaluation (log scale).
    Label = reach-rate: % of trials that ever reach P-loss ≤ 0.05 within the budget.

    > **Important:** `sec→threshold` is averaged only over trials that *reach*
    > the threshold — a method with a low reach-rate looks misleadingly fast.
        """),
        mo.image(str(FIG / "04_walltime_mseval.png")),
        mo.md(r"""
    | Synth | GD ms/eval | RS ms/eval | CMA-ES ms/eval | BO ms/eval |
    |---|---|---|---|---|
    | bandpass_noise | **2 212** | 22 | 25 | 208 |
    | am_noise | 55 | 283 | 286 | 564 |
    | add_sinesaw | 24 | 19 | 19 | 231 |

    **Key findings:**

    - GD's per-step cost is **real, not a compile artifact** (probe: 2.13 s/step,
      only 4.8 s fixed setup).  Each step runs a full differentiable
      forward + backward pass over 44 100 samples — ~90× a black-box render on
      the noise synth.
    - On `bandpass_noise`: CMA-ES solves **96 %** of targets in **1.0 s**;
      GD solves only **56 %** and takes **28.8 s** (~29× slower wall-clock).
    - On `am_noise` all gradient-free methods cost ~283 ms/eval — the numpy
      DTW loss dominates, not the render.  GD is cheap here (JAX DTW, 55 ms)
      but only solves **14 %** of targets.
    - GD's structural requirement — a differentiable synthesizer — rules it
      out for VSTs and most real-world instruments regardless of speed.
        """),
    ])
    return


@app.cell
def _(FIG, mo):
    mo.vstack([
        mo.md(r"""
    ---
    ## 7. Non-identifiability of `add_sinesaw`

    Scatter of audio-loss vs P-loss across all evaluated points (CMA-ES,
    50 trials sampled).  A perfectly informative loss would show a strong
    positive correlation; a flat scatter means the loss carries almost no
    information about the true parameters.
        """),
        mo.image(str(FIG / "06_identifiability_scatter.png")),
        mo.md(r"""
    **Mechanism.**
    `add_sinesaw = sineOsc(sine_freq) + sawOsc(saw_freq)`.
    A sawtooth fills the spectrum with a dense harmonic comb; the sine is a
    single peak among dozens of harmonics.  Under magnitude-spectrogram MSE
    (SIMSE_Spec):

    - Different saw frequencies produce overlapping combs → nearly identical spectra.
    - Moving `sine_freq` changes one bin in a dense spectrogram → negligible MSE change.

    **Grid scan result (41 × 41 over the parameter box):**
    76 % of parameter space scores within 1.5× of the global loss minimum.
    A point at P-loss = 0.79 (near the opposite corner) scores within 4 % of
    the best loss.

    This surface is kept deliberately as a **stress test**: the question is
    which method returns the least-wrong parameters *despite* the deceptive
    loss.
        """),
    ])
    return


@app.cell
def _(FIG, mo):
    mo.vstack([
        mo.md(r"""
    ---
    ## 8. Learned amortized inverse model (preview)

    A small MLP trained offline on 15 000 uniformly-sampled `(params → audio)`
    pairs to predict parameters directly from a spectral feature vector
    (512-d log-magnitude STFT, scipy, no librosa).  Evaluated **zero-shot**
    (0 audio evaluations at deploy time) on the exact same 200 benchmark targets.

    | Synth | Learned (0 eval) | CMA-ES (200) | BO (200) | GD (200) |
    |---|---|---|---|---|
    | bandpass_noise | 0.015 | **0.001** | 0.041 | 0.087 |
    | am_noise | 0.034 | **0.008** | 0.014 | 0.488 |
    | `add_sinesaw` | **0.045** | 0.336 | 0.197 | 0.520 |
        """),
        mo.image(str(FIG / "05_learned_vs_optimizers.png")),
        mo.md(r"""
    **Headline result:**

    - On identifiable synths the learned model is *competitive with the best
      optimizer at 200 evals* — at **zero evaluations** (amortized efficiency).
    - On the non-identifiable `add_sinesaw` it **beats every optimizer ~4×**
      (0.045 vs BO 0.197) — because it uses a learned parameter→sound prior
      rather than the deceptive instantaneous loss.

    **Why this works on the deceptive surface:**
    The model is trained with *direct supervision on parameters* (not audio
    loss), so it learns the true prior over which spectral features correspond
    to which parameters — information the audio loss cannot provide at test time.

    **Efficiency profile:** training is paid once (~15 min, 15 k renders);
    deployment costs a single forward pass (~1 ms).  This is a fundamentally
    different efficiency profile from cold-start search.

    **Caveat:** this is an MLP regressor (B1 in the design plan).  Non-identifiable
    synths have a one-to-many mapping; the MLP returns the mean of valid parameter
    modes, which explains its moderate performance on add_sinesaw despite the win
    — a multi-modal predictor (B3) would push further.
        """),
    ])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(r"""
    ---
    ## 9. Summary

    ### Returned P-loss (median) — what each method deploys

    | Synth | GD | RS | CMA-ES | BO | Learned |
    |---|---|---|---|---|---|
    | bandpass_noise | 0.087 | 0.060 | **0.001** | 0.041 | 0.015 |
    | am_noise | 0.488 | 0.049 | **0.008** | 0.014 | 0.034 |
    | add_sinesaw | 0.520 | 0.152 | 0.336 | 0.197 | **0.045** |

    ### Method × landscape suitability

    | Landscape | Best method | Why others fail |
    |---|---|---|
    | Smooth, identifiable (`bandpass`) | **CMA-ES** | GD slow + high failure tail; BO over-explores |
    | Moderate (`am_noise`) | **CMA-ES / BO** | GD gradient too weak; RS needs more evals |
    | Flat / non-identifiable (`add_sinesaw`) | **Learned** (0-eval), then **BO** | CMA-ES commits to wrong basin (gap 0.298); GD completely blind |

    ### What makes GD competitive (and where it isn't)

    GD is the only method requiring a differentiable synthesizer.  It is
    competitive on `bandpass_noise` in terms of *visited* P-loss (0.025) but:
    - Is ~29× slower wall-clock per solution on `bandpass_noise`
    - Completely fails on `am_noise` (the DTW gradient is too weak at lr = 0.045)
    - Returns the worst parameters on 2/3 synths (returned 0.488, 0.520)

    The structural cost — differentiability — rules it out for most real instruments.
        """),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## 10. Interactive explorer
    """)
    return


@app.cell
def _(RES, load_all_results):
    all_results = load_all_results(RES)
    synths = sorted(all_results)
    return all_results, synths


@app.cell
def _(mo, synths):
    synth_sel = mo.ui.dropdown(options=synths, value=synths[0], label="Synth")
    synth_sel
    return (synth_sel,)


@app.cell
def _(METHODS, RES, mo, np, pd, pickle, synth_sel):
    def _build_stats(synth):
        rows = []
        for m in METHODS:
            f = RES / f"{synth}_{m}.pkl"
            if not f.exists():
                continue
            tr = pickle.load(open(f, "rb"))["trials"]
            ret = np.median([np.array(t["history_p_loss"])[int(np.argmin(t["history_audio_loss"]))] for t in tr])
            vis = np.median([np.min(t["history_p_loss"]) for t in tr])
            dur = np.median([t.get("duration_s", np.nan) for t in tr])
            rows.append({
                "method": m, "n": len(tr),
                "returned (median)": round(float(ret), 4),
                "visited (median)": round(float(vis), 4),
                "deception gap": round(float(ret - vis), 4),
                "duration_s (median)": round(float(dur), 1),
            })
        lp = RES / "learned_results.pkl"
        if lp.exists():
            L = pickle.load(open(lp, "rb")).get(synth)
            if L:
                rows.append({
                    "method": "Learned (0-eval)", "n": 200,
                    "returned (median)": round(float(L["median"]), 4),
                    "visited (median)": round(float(L["median"]), 4),
                    "deception gap": 0.0,
                    "duration_s (median)": 0.0,
                })
        return pd.DataFrame(rows)

    _out = mo.vstack([
        mo.md("### Per-synth stats"),
        mo.ui.table(_build_stats(synth_sel.value), selection=None),
    ]) if synth_sel.value else mo.md("_no data_")
    _out
    return


@app.cell
def _(all_results, mo, synth_sel):
    _avail = [m for m in ["GD", "RandomSearch", "CMA-ES", "BO"]
              if m in all_results.get(synth_sel.value, {})]
    method_sel = mo.ui.dropdown(options=_avail, value=_avail[0] if _avail else None, label="Method")
    trial_sel  = mo.ui.number(start=0, stop=199, step=1, value=0, label="Trial index")
    _out = mo.hstack([method_sel, trial_sel])
    _out
    return method_sel, trial_sel


@app.cell
def _(
    all_results,
    build_trial_context,
    compute_surface,
    extract_trajectory,
    method_sel,
    mo,
    plot_full_trajectory,
    plt,
    synth_sel,
    trial_sel,
):
    if not (synth_sel.value and method_sel.value):
        _out = mo.md("_select a synth and method above_")
    else:
        _ctx  = build_trial_context(all_results, synth_sel.value, method_sel.value, trial_sel.value)
        _traj = extract_trajectory(_ctx)
        if _traj["trajectory"].size == 0 or _traj["trajectory"].shape[1] != 2:
            _out = mo.md("_no 2D trajectory available_")
        else:
            _xx, _yy, _surf, _lab = compute_surface("P-Loss surface", 31, _ctx, _traj)
            _fig = plot_full_trajectory(_traj, _ctx, _xx, _yy, _surf, _lab)
            _out = mo.as_html(_fig)
            plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 11. Notes & caveats

    - **Wall-clock** measurements (§5) are from a single controlled run on a
      128-core shared server under varying background load; the relative ordering
      is reliable but absolute values need a dedicated clean machine.
    - **BO per-eval cost** is underestimated by the two-budget slope at low budget
      (GP refit is O(n³)); re-measure at budget ≈ 200 before quoting.
    - **am_noise wall-clock** is loss-bound, not render-bound (~283 ms/eval for all
      gradient-free methods); using the JAX DTW for evaluation would remove this.
    - **All synths are 2-dimensional.** Scaling to higher-d synths is needed
      before making general claims about method complexity.
    - **Learned model (§8)** is a point-estimate MLP regressor.  The non-identifiable
      add_sinesaw has multiple valid parameter modes; a mixture-density network
      would handle this more faithfully.
    - **RandomSearch** is now uniform sampling over [0,1]^d — the canonical
      statistical baseline.  The earlier implementation was a random walk that
      ignored the objective and has been replaced.
    """)
    return


if __name__ == "__main__":
    app.run()
