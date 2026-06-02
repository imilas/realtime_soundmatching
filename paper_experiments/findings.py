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

    RES = Path(__file__).parent / "results"
    FIG = RES / "figures"
    METHODS = ["GD", "RandomSearch", "CMA-ES", "BO"]
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
    _out = mo.md(
        """
        # Sound-matching optimizer study — interactive findings

        Benchmark of four search methods (GD, RandomSearch, CMA-ES, BO) + a
        learned amortized model on three synthesizers chosen as **loss-landscape archetypes**
        (`bandpass_noise` smooth/identifiable, `am_noise` moderate, `add_sinesaw`
        flat/non-identifiable). Budget 200 evals, matched seeds.

        Scroll for the headline results, then use the **interactive stats** and
        **trajectory explorer** below. Full write-up: `WEEKEND_REPORT.md`.
        """
    )
    _out
    return


@app.cell
def _(FIG, mo):
    _out = mo.vstack([
        mo.md(
            "## Headline — the learned model beats search where it matters\n\n"
            "Zero-shot (0 audio evals) the learned inverse model beats BO and GD "
            "everywhere and **beats every optimizer ~4× on the non-identifiable "
            "`add_sinesaw`** (0.045 vs BO 0.197 at 200 evals) — it uses a learned "
            "param→sound prior instead of the deceptive loss."
        ),
        mo.image(str(FIG / "05_learned_vs_optimizers.png")),
    ])
    _out
    return


@app.cell
def _(FIG, mo):
    _out = mo.vstack([
        mo.md(
            "## The deception gap (why optimizers fail)\n\n"
            "**Visited** = best params ever sampled (oracle). **Returned** = what "
            "the method deploys (argmin audio loss). On `add_sinesaw` every method "
            "*visits* good params but *returns* bad ones — and CMA-ES is the **most** "
            "deceived because it commits to one (wrong) basin, while BO hedges."
        ),
        mo.image(str(FIG / "02_returned_vs_visited.png")),
    ])
    _out
    return


@app.cell
def _(mo):
    _out = mo.md("## Interactive — per-synth stats")
    _out
    return


@app.cell
def _(RES, load_all_results):
    all_results = load_all_results(RES)
    synths = sorted(all_results)
    return all_results, synths


@app.cell
def _(mo, synths):
    synth_pick = mo.ui.dropdown(options=synths, value=synths[0] if synths else None, label="Synth")
    synth_pick
    return (synth_pick,)


@app.cell
def _(METHODS, RES, mo, np, pd, pickle, synth_pick):
    def _stats(synth):
        rows = []
        for m in METHODS:
            f = RES / f"{synth}_{m}.pkl"
            if not f.exists():
                continue
            tr = pickle.load(open(f, "rb"))["trials"]
            ret = np.median([np.array(t["history_p_loss"])[int(np.argmin(t["history_audio_loss"]))] for t in tr])
            vis = np.median([np.min(t["history_p_loss"]) for t in tr])
            rows.append({"method": m, "n": len(tr),
                         "returned": round(float(ret), 3), "visited": round(float(vis), 3),
                         "deception_gap": round(float(ret - vis), 3)})
        # learned
        lp = RES / "learned_results.pkl"
        if lp.exists():
            L = pickle.load(open(lp, "rb")).get(synth)
            if L:
                rows.append({"method": "Learned (0-eval)", "n": 200,
                             "returned": round(float(L["median"]), 3),
                             "visited": round(float(L["median"]), 3), "deception_gap": 0.0})
        return pd.DataFrame(rows)

    _out = mo.ui.table(_stats(synth_pick.value), selection=None) if synth_pick.value else mo.md("_no data_")
    _out
    return


@app.cell
def _(mo):
    _out = mo.md(
        "## Interactive — 2D update trajectory\n\n"
        "Pick a method and trial to watch the search move over the P-loss surface "
        "(★ true params, ■ init, ✖ best). Confirms *how* each method behaves."
    )
    _out
    return


@app.cell
def _(all_results, mo, synth_pick):
    _avail = sorted(all_results.get(synth_pick.value, {})) if synth_pick.value else []
    method_pick = mo.ui.dropdown(options=_avail, value=_avail[0] if _avail else None, label="Method")
    trial_pick = mo.ui.number(start=0, stop=199, step=1, value=0, label="Trial")
    _out = mo.hstack([method_pick, trial_pick])
    _out
    return method_pick, trial_pick


@app.cell
def _(
    all_results,
    build_trial_context,
    compute_surface,
    extract_trajectory,
    method_pick,
    mo,
    plot_full_trajectory,
    plt,
    synth_pick,
    trial_pick,
):
    if not (synth_pick.value and method_pick.value):
        _out = mo.md("_select a synth and method_")
    else:
        _ctx = build_trial_context(all_results, synth_pick.value, method_pick.value, trial_pick.value)
        _traj = extract_trajectory(_ctx)
        if _traj["trajectory"].size == 0 or _traj["trajectory"].shape[1] != 2:
            _out = mo.md("_no 2D trajectory for this selection_")
        else:
            _xx, _yy, _surf, _lab = compute_surface("P-Loss surface", 31, _ctx, _traj)
            _fig = plot_full_trajectory(_traj, _ctx, _xx, _yy, _surf, _lab)
            _out = mo.as_html(_fig)
            plt.close(_fig)
    _out
    return


@app.cell
def _(FIG, mo):
    _out = mo.vstack([
        mo.md("## Sample efficiency & wall-clock"),
        mo.image(str(FIG / "03_learning_curves.png")),
        mo.md(
            "Controlled wall-clock (ms/eval bar, reach-rate label): on identifiable "
            "bandpass, CMA-ES solves 96% of targets in 1.0 s vs **GD 56% in 28.8 s "
            "(~29× slower)**. am_noise is loss-bound (~283 ms/eval for all)."
        ),
        mo.image(str(FIG / "04_walltime_mseval.png")),
    ])
    _out
    return


@app.cell
def _(FIG, mo):
    _out = mo.vstack([
        mo.md("## Supporting figures"),
        mo.md("**Final accuracy** (best P-loss per method, log scale):"),
        mo.image(str(FIG / "01_boxplots_bestploss.png")),
        mo.md("**Non-identifiability** (audio-loss vs P-loss; flat ⇒ uninformative):"),
        mo.image(str(FIG / "06_identifiability_scatter.png")),
    ])
    _out
    return


@app.cell
def _(RES, mo):
    _p = RES / "STATS_SUMMARY.md"
    _out = mo.md(_p.read_text()) if _p.exists() else mo.md("_run make_figures.py for STATS_SUMMARY.md_")
    _out
    return


if __name__ == "__main__":
    app.run()
