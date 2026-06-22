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
