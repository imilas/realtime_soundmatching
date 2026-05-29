import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import marimo as mo
    import matplotlib.pyplot as plt

    from utils.notebooks.viz_funcs import (
        METHOD_ORDER,
        load_data,
        keep_last_ql_pct,
        plot_boxplots,
        plot_ql_learning,
        compute_summary_table,
    )

    RESULTS_DIR = Path("paper_experiments/results")
    return (
        RESULTS_DIR,
        compute_summary_table,
        keep_last_ql_pct,
        load_data,
        mo,
        plot_boxplots,
        plot_ql_learning,
        plt,
    )


@app.cell
def _(RESULTS_DIR, load_data):
    df = load_data(RESULTS_DIR)
    synths_available = sorted(df["program"].unique()) if not df.empty else []
    methods_available = sorted(df["method"].unique()) if not df.empty else []
    print(f"Loaded {len(df)} trials | synths: {synths_available} | methods: {methods_available}")
    return df, methods_available, synths_available


@app.cell
def _(methods_available, mo, synths_available):
    synth_picker = mo.ui.multiselect(options=synths_available, value=synths_available, label="Synths")
    method_picker = mo.ui.multiselect(options=methods_available, value=methods_available, label="Methods")
    ql_tail_pct = mo.ui.slider(start=1, stop=100, step=1, value=100, label="QL last %")
    _out = mo.hstack([synth_picker, method_picker, ql_tail_pct])
    _out
    return method_picker, ql_tail_pct, synth_picker


@app.cell
def _(df, keep_last_ql_pct, method_picker, ql_tail_pct, synth_picker):
    selected = df[df["program"].isin(synth_picker.value) & df["method"].isin(method_picker.value)]
    filtered = keep_last_ql_pct(selected, ql_tail_pct.value)
    return filtered, selected


@app.cell
def _(filtered, mo, plot_boxplots, plt):
    if filtered.empty:
        _out = mo.md("_No data for current selection._")
    else:
        _fig = plot_boxplots(filtered)
        _out = mo.as_html(_fig)
        plt.close(_fig)
    _out
    return


@app.cell
def _(mo, plot_ql_learning, plt, selected):
    _fig = plot_ql_learning(selected)
    if _fig is None:
        _out = mo.md("_No QL trials in current selection._")
    else:
        _out = mo.as_html(_fig)
        plt.close(_fig)
    _out
    return


@app.cell
def _(compute_summary_table, filtered, mo):
    if filtered.empty:
        _out = mo.md("_No data._")
    else:
        _out = mo.ui.table(compute_summary_table(filtered), selection=None)
    _out
    return


if __name__ == "__main__":
    app.run()
