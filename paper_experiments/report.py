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
    METHODS = ["GD", "RandomSearch", "CMA-ES", "LES"]
    SYNTHS  = ["bandpass_noise", "am_noise", "add_sinesaw"]
    COLORS  = {"GD": "black", "RandomSearch": "#ff7f0e", "CMA-ES": "#2ca02c", "LES": "#9467bd"}
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
def _(FIG, mo):
    mo.image(str(FIG / "01_boxplots_bestploss.png"))
    return


@app.cell
def _(FIG, mo):
    mo.image(str(FIG / "02_returned_vs_visited.png"))
    return


@app.cell
def _(FIG, mo):
    mo.image(str(FIG / "03_learning_curves.png"))
    return


@app.cell
def _(FIG, mo):
    mo.image(str(FIG / "07_returned_curves.png"))
    return


@app.cell
def _(FIG, mo):
    mo.image(str(FIG / "04_walltime_mseval.png"))
    return


@app.cell
def _(FIG, mo):
    mo.image(str(FIG / "05_clap_walltime.png"))
    return


@app.cell
def _(FIG, mo):
    mo.image(str(FIG / "06_identifiability_scatter.png"))
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
def _(METHODS, all_results, mo, np, pd, synth_sel):
    def _build_stats(synth):
        rows = []
        for m in METHODS:
            data = all_results.get(synth, {}).get(m)
            if data is None:
                continue
            tr = data["trials"]
            ret = np.median([t["history_p_loss"][-1] for t in tr])
            vis = np.median([np.min(t["history_p_loss"]) for t in tr])
            dur = np.median([t.get("duration_s", np.nan) for t in tr])
            rows.append({
                "method": m, "n": len(tr),
                "final (median)": round(float(ret), 4),
                "visited (median)": round(float(vis), 4),
                "final vs visited gap": round(float(ret - vis), 4),
                "duration_s (median)": round(float(dur), 1),
            })
        return pd.DataFrame(rows)

    _out = mo.vstack([
        mo.md("### Per-synth stats"),
        mo.ui.table(_build_stats(synth_sel.value), selection=None),
    ]) if synth_sel.value else mo.md("_no data_")
    _out
    return


@app.cell
def _(METHODS, all_results, mo, synth_sel):
    _avail = [m for m in METHODS
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
def _(RES, np, pickle):
    import re as _re
    from scipy.stats import mannwhitneyu as _mannwhitneyu

    ML_CORE_LOSSES = ["SIMSE_Spec", "JTFS", "DTW_Envelope", "CLAP"]
    ML_METHODS = ["GD", "RandomSearch", "CMA-ES", "LES"]
    ML_SYNTHS = [
        "bandpass_noise", "am_noise", "add_sinesaw",
        "sine_saw", "sine_mod_saw", "sine_mod_sine",
        "chirplet", "chirplet_pulse",
    ]

    def _slug(s):
        return _re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")

    def _returned_p_loss_ml(trial, method):
        ploss = np.asarray(trial.get("history_p_loss", []), dtype=float)
        if len(ploss) == 0:
            return float("nan")
        if method == "GD":
            # GD converges to a point; use the final gradient step.
            return float(ploss[-1])
        audio = np.asarray(trial.get("history_audio_loss", []), dtype=float)
        if len(audio) == 0:
            return float("nan")
        return float(ploss[int(np.nanargmin(audio))])

    def load_scores_ml(synth, loss, method):
        path = RES / f"{synth}_{_slug(loss)}_{method}.pkl"
        if not path.exists():
            return []
        with path.open("rb") as fh:
            trials = pickle.load(fh).get("trials", [])
        return [s for s in (_returned_p_loss_ml(t, method) for t in trials) if np.isfinite(s)]

    def _cliffs_delta(a, b):
        a, b = np.asarray(a), np.asarray(b)
        count = sum(np.sum(ai > b) - np.sum(ai < b) for ai in a)
        return count / (len(a) * len(b))

    def npsk_ranks(groups_dict, alpha=0.05, negligible_delta=0.147):
        """NPSK: Scott-Knott with Cliff's delta effect-size gate on raw scores.

        Sorts by group mean (ascending = lower P-loss = better).
        Splits only when Mann-Whitney p < alpha AND |Cliff's delta| >= negligible_delta.
        Returns {name: int rank}, rank 1 = best.
        """
        valid = {}
        for name, scores in groups_dict.items():
            arr = np.asarray(scores, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                valid[name] = arr
        if len(valid) < 2:
            return {name: 1 for name in valid}

        sorted_names = sorted(valid, key=lambda g: valid[g].mean())

        def sk_split(names):
            if len(names) <= 1:
                return [list(names)]
            all_vals = np.concatenate([valid[n] for n in names])
            grand_mean = all_vals.mean()
            best_bss, best_i = -1.0, 1
            for i in range(1, len(names)):
                lv = np.concatenate([valid[n] for n in names[:i]])
                rv = np.concatenate([valid[n] for n in names[i:]])
                bss = (len(lv) * (lv.mean() - grand_mean) ** 2
                       + len(rv) * (rv.mean() - grand_mean) ** 2)
                if bss > best_bss:
                    best_bss, best_i = bss, i
            lv = np.concatenate([valid[n] for n in names[:best_i]])
            rv = np.concatenate([valid[n] for n in names[best_i:]])
            _, p = _mannwhitneyu(lv, rv, alternative="two-sided")
            delta = _cliffs_delta(lv, rv)
            if p < alpha and abs(delta) >= negligible_delta:
                return sk_split(names[:best_i]) + sk_split(names[best_i:])
            return [list(names)]

        partitions = sk_split(sorted_names)
        return {n: r for r, grp in enumerate(partitions, 1) for n in grp}

    return ML_CORE_LOSSES, ML_METHODS, ML_SYNTHS, load_scores_ml, npsk_ranks


@app.cell
def _(ML_CORE_LOSSES, ML_METHODS, ML_SYNTHS, load_scores_ml, mo, np, pd):
    def _build_median_table(synth):
        rows = []
        for loss in ML_CORE_LOSSES:
            row = {"Loss": loss}
            for method in ML_METHODS:
                scores = load_scores_ml(synth, loss, method)
                row[method] = float(np.median(scores)) if scores else float("nan")
            rows.append(row)
        df = pd.DataFrame(rows).set_index("Loss")

        def _fmt_row(row):
            finite = row.dropna()
            if finite.empty:
                return row.map(lambda v: "—" if np.isnan(v) else f"{v:.4f}")
            best_val = finite.min()
            return row.map(
                lambda v: "—" if np.isnan(v)
                else f"**{v:.4f}**" if v == best_val
                else f"{v:.4f}"
            )

        formatted = df.apply(_fmt_row, axis=1)
        formatted.index.name = "Loss \\ Method"
        return formatted.reset_index()

    _tables = []
    for _synth in ML_SYNTHS:
        _t = _build_median_table(_synth)
        if _t.iloc[:, 1:].apply(lambda col: col != "—").any().any():
            _tables.append(mo.vstack([
                mo.md(f"### `{_synth}`"),
                mo.ui.table(_t, label="Median returned P-loss", page_size=len(_t)),
            ]))

    mo.vstack(_tables)
    return


@app.cell
def _(
    ML_CORE_LOSSES,
    ML_METHODS,
    ML_SYNTHS,
    load_scores_ml,
    mo,
    npsk_ranks,
    pd,
):
    def _build_npsk_table(synth):
        rows = []
        for loss in ML_CORE_LOSSES:
            groups = {m: load_scores_ml(synth, loss, m) for m in ML_METHODS}
            groups = {m: s for m, s in groups.items() if s}
            if len(groups) < 2:
                continue
            ranks = npsk_ranks(groups)
            row = {"Loss": loss}
            for method in ML_METHODS:
                row[method] = int(ranks[method]) if method in ranks else "—"
            rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    _npsk_tables = []
    for _synth in ML_SYNTHS:
        _df = _build_npsk_table(_synth)
        if len(_df):
            _npsk_tables.append(mo.vstack([
                mo.md(f"### `{_synth}`"),
                mo.ui.table(_df, label="NPSK rank (1 = best)", page_size=len(_df)),
            ]))

    mo.vstack(_npsk_tables)
    return


@app.cell
def _(ML_METHODS, load_scores_ml, mo, npsk_ranks, pd):
    _SYNTH_LOSS_CANONICAL = {
        "bandpass_noise": "SIMSE_Spec",
        "am_noise":       "DTW_Envelope",
        "add_sinesaw":    "SIMSE_Spec",
        "sine_saw":       "JTFS",
        "sine_mod_saw":   "JTFS",
        "sine_mod_sine":  "JTFS",
        "chirplet":       "JTFS",
        "chirplet_pulse": "DTW_Envelope",
    }

    _summary_rows = {m: {} for m in ML_METHODS}
    for _synth, _loss in _SYNTH_LOSS_CANONICAL.items():
        _groups = {m: load_scores_ml(_synth, _loss, m) for m in ML_METHODS}
        _groups = {m: s for m, s in _groups.items() if s}
        if len(_groups) < 2:
            continue
        _ranks = npsk_ranks(_groups)
        for _m in ML_METHODS:
            _summary_rows[_m][_synth] = _ranks.get(_m, "—")

    _synth_cols = list(_SYNTH_LOSS_CANONICAL)
    _summary_df = pd.DataFrame(
        [{**{"Method": m}, **{s: _summary_rows[m].get(s, "—") for s in _synth_cols}}
         for m in ML_METHODS]
    )

    mo.vstack([
        mo.md("**NPSK ranks — methods × synthesizers (canonical loss)**"),
        mo.ui.table(_summary_df, label="Rank 1 = best; ties = statistically indistinguishable", page_size=len(_summary_df)),
    ])
    return


@app.cell
def _(ML_METHODS, ML_SYNTHS, load_scores_ml, mo, np, npsk_ranks, plt):
    _SYNTH_LOSS_CANONICAL = {
        "bandpass_noise": "SIMSE_Spec",
        "am_noise":       "DTW_Envelope",
        "add_sinesaw":    "SIMSE_Spec",
        "sine_saw":       "JTFS",
        "sine_mod_saw":   "JTFS",
        "sine_mod_sine":  "JTFS",
        "chirplet":       "JTFS",
        "chirplet_pulse": "DTW_Envelope",
    }

    _RANK_COLORS = {1: "#2ca02c", 2: "#ff7f0e", 3: "#d62728", 4: "#7f0000"}

    def _bootstrap_medians(scores, n_boot=1000, rng_seed=0):
        rng = np.random.default_rng(rng_seed)
        arr = np.asarray(scores, float)
        return np.array([np.median(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)])

    _ncols = 4
    _nrows = (len(ML_SYNTHS) + _ncols - 1) // _ncols
    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(14, 3.5 * _nrows), squeeze=False)

    for _idx, _synth in enumerate(ML_SYNTHS):
        _ax = _axes[_idx // _ncols][_idx % _ncols]
        _loss = _SYNTH_LOSS_CANONICAL[_synth]
        _groups = {m: load_scores_ml(_synth, _loss, m) for m in ML_METHODS}
        _groups = {m: s for m, s in _groups.items() if s}
        if len(_groups) < 2:
            _ax.set_visible(False)
            continue
        _ranks = npsk_ranks(_groups)

        _violin_data, _positions, _colors, _labels = [], [], [], []
        for _pi, _m in enumerate([m for m in ML_METHODS if m in _groups]):
            _boots = _bootstrap_medians(_groups[_m])
            _violin_data.append(_boots)
            _positions.append(_pi)
            _colors.append(_RANK_COLORS.get(_ranks.get(_m, 4), "#999999"))
            _labels.append(_m)

        _vp = _ax.violinplot(_violin_data, positions=_positions, showmedians=True, widths=0.7)
        for _body, _c in zip(_vp["bodies"], _colors):
            _body.set_facecolor(_c)
            _body.set_alpha(0.75)
        for _part in ("cmedians", "cmins", "cmaxes", "cbars"):
            _vp[_part].set_color("black")
            _vp[_part].set_linewidth(0.8)

        _ax.set_xticks(_positions)
        _ax.set_xticklabels(_labels, fontsize=8)
        _ax.set_title(f"{_synth}\n({_loss})", fontsize=8)
        _ax.set_ylabel("bootstrapped median P-loss", fontsize=7)
        _ax.tick_params(axis="y", labelsize=7)

    # Hide unused axes
    for _idx in range(len(ML_SYNTHS), _nrows * _ncols):
        _axes[_idx // _ncols][_idx % _ncols].set_visible(False)

    _fig.suptitle(
        "Bootstrapped median returned P-loss  |  color = NPSK rank  "
        "(green=1st, orange=2nd, red=3rd, darkred=4th)",
        fontsize=9, y=1.01,
    )
    _fig.tight_layout()
    _out = mo.as_html(_fig)
    plt.close(_fig)
    _out
    return


if __name__ == "__main__":
    app.run()
