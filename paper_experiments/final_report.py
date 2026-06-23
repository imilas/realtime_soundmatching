import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import marimo as mo
    import pandas as pd

    from paper_experiments.make_verification_report import (
        METHODS, LOSSES, LOSSES_ALL, SYNTHS, SYNTH_LABELS, PUBLISHED,
        load_trials, extract_scores, bootstrap_medians, npsk_ranks,
    )

    RES = Path(__file__).parent / "results"
    FIG = RES / "figures"
    return (
        FIG,
        LOSSES_ALL,
        METHODS,
        PUBLISHED,
        SYNTHS,
        SYNTH_LABELS,
        bootstrap_medians,
        extract_scores,
        load_trials,
        mo,
        npsk_ranks,
        pd,
    )


@app.cell
def _(SYNTHS, SYNTH_LABELS, mo):
    # Pick which synths to display. Defaults to all; the tables/curves below
    # react to this selection.
    synth_select = mo.ui.multiselect(
        options={SYNTH_LABELS[s]: s for s in SYNTHS},
        value=[SYNTH_LABELS[s] for s in SYNTHS],   # value = option keys (labels); .value returns the synth ids
        label="Synths to show",
    )
    mo.vstack([mo.md("### Select synths"), synth_select])
    return (synth_select,)


@app.cell
def _(
    LOSSES_ALL,
    METHODS,
    PUBLISHED,
    SYNTHS,
    SYNTH_LABELS,
    bootstrap_medians,
    extract_scores,
    load_trials,
    mo,
    npsk_ranks,
    pd,
    synth_select,
):
    def _rank_or_dash(r):
        return r if isinstance(r, int) else "—"

    # honor the selector; keep canonical SYNTHS order; fall back to all if empty
    _selected = [s for s in SYNTHS if s in synth_select.value] or list(SYNTHS)
    _tables = []
    for _synth in _selected:
        _rows = []
        _pub_row = {"": "Published"}
        for _loss in LOSSES_ALL:
            _pub_row[_loss] = PUBLISHED[_synth].get(_loss, "—")
        _rows.append(_pub_row)

        for _method in METHODS:
            _groups = {}
            for _loss in LOSSES_ALL:
                _trials = load_trials(_synth, _loss, _method)
                _returned, _ = extract_scores(_trials, _method)
                if len(_returned) >= 2:
                    _groups[_loss] = bootstrap_medians(_returned)
            _ranks = npsk_ranks(_groups) if len(_groups) >= 2 else {}
            _row = {"": _method}
            for _loss in LOSSES_ALL:
                _row[_loss] = _rank_or_dash(_ranks.get(_loss, "—"))
            _rows.append(_row)

        _df = pd.DataFrame(_rows).set_index("")
        _tables.append(
            mo.vstack([
                mo.md(f"### {SYNTH_LABELS[_synth]} (`{_synth}`)"),
                mo.ui.table(_df.reset_index(), selection=None, label=SYNTH_LABELS[_synth]),
            ])
        )

    mo.vstack(_tables)
    return


@app.cell
def _(
    LOSSES_ALL,
    METHODS,
    SYNTHS,
    SYNTH_LABELS,
    load_trials,
    mo,
    synth_select,
):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io

    _METHOD_COLORS = {
        "GD": "#e6194b",
        "CMA-ES": "#3cb44b",
        "LES": "#4363d8",
        "RandomSearch": "#f58231",
    }

    _selected_synths = [s for s in SYNTHS if s in synth_select.value] or list(SYNTHS)

    # One row per synth, one column per loss
    _losses = LOSSES_ALL
    _n_rows = len(_selected_synths)
    _n_cols = len(_losses)

    _fig, _axes = plt.subplots(
        _n_rows, _n_cols,
        figsize=(4.5 * _n_cols, 3.5 * _n_rows),
        squeeze=False,
    )

    for _ri, _synth in enumerate(_selected_synths):
        for _ci, _loss in enumerate(_losses):
            _ax = _axes[_ri][_ci]
            _any = False
            for _method in METHODS:
                _trials = load_trials(_synth, _loss, _method)
                if not _trials:
                    continue
                # Collect audio-loss curves; pad/trim to the shortest length
                _curves = []
                for _t in _trials:
                    _al = np.asarray(_t.get("history_audio_loss", []), dtype=float)
                    if len(_al) > 0:
                        _curves.append(_al)
                if not _curves:
                    continue
                _min_len = min(len(c) for c in _curves)
                _mat = np.stack([c[:_min_len] for c in _curves])  # (n_trials, steps)
                _median = np.median(_mat, axis=0)
                _steps = np.arange(1, _min_len + 1)
                _ax.plot(_steps, _median, label=_method,
                         color=_METHOD_COLORS.get(_method, "grey"), linewidth=1.5)
                _any = True

            if _ri == 0:
                _ax.set_title(_loss, fontsize=10)
            if _ci == 0:
                _ax.set_ylabel(SYNTH_LABELS[_synth], fontsize=9)
            _ax.set_xlabel("eval step" if _ri == _n_rows - 1 else "")
            _ax.tick_params(labelsize=8)
            if _any:
                _ax.set_yscale("symlog", linthresh=1e-6)

    # Shared legend on the last axes
    _handles, _labels = _axes[0][0].get_legend_handles_labels()
    _fig.legend(_handles, _labels, loc="upper right", fontsize=9, framealpha=0.8)
    _fig.suptitle("Median audio-loss curves per method", fontsize=12, y=1.01)
    _fig.tight_layout()

    _buf = io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(_fig)
    _buf.seek(0)
    mo.image(_buf.read())
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
    mo.image(str(FIG / "06_identifiability_scatter.png"))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
