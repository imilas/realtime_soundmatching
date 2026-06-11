import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import pickle
    import re
    import marimo as mo
    import numpy as np
    import pandas as pd
    from datetime import datetime

    RES = Path(__file__).parent / "results"
    return RES, datetime, mo, np, pd, pickle, re


@app.cell
def _(RES, datetime, np, pd, pickle, re):
    # Scan all {synth}_{loss}_{method}.pkl files in results/
    _PAT = re.compile(
        r"^(?P<synth>.+?)_(?P<loss>SIMSE_Spec|L1_Spec|JTFS|DTW_Envelope)"
        r"_(?P<method>GD|CMA-ES|RandomSearch|LES|CMA-ES-evosax)\.pkl$"
    )

    _rows = []
    for _f in sorted(RES.glob("*.pkl")):
        _m = _PAT.match(_f.name)
        if not _m:
            continue
        _synth, _loss, _method = _m.group("synth"), _m.group("loss"), _m.group("method")
        try:
            _d = pickle.load(open(_f, "rb"))
            _trials = _d.get("trials", [])
        except Exception:
            _trials = []

        _n = len(_trials)
        if _n > 0:
            _budget = int(_trials[0].get("eval_budget", len(_trials[0].get("history_p_loss", []))))
            _dur = [t.get("duration_s") for t in _trials if t.get("duration_s") is not None]
            _dur_med = f"{np.median(_dur):.1f}s" if _dur else "—"
        else:
            _budget = 0
            _dur_med = "—"

        _mtime = datetime.fromtimestamp(_f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        _rows.append({
            "synth": _synth,
            "loss": _loss,
            "method": _method,
            "n trials": _n,
            "budget": _budget,
            "med duration/trial": _dur_med,
            "last modified": _mtime,
        })

    status_df = pd.DataFrame(_rows)
    return (status_df,)


@app.cell
def _(mo, status_df):
    mo.md(f"""
    # Experiment status\n\n"
        f"{len(status_df)} result files found.  "
        f"**{int((status_df['n trials'] >= 200).sum())}** cells complete (≥ 200 trials), "
        f"**{int((status_df['n trials'] < 200).sum())}** partial or empty.
    """)
    return


@app.cell
def _(mo, status_df):
    # Filter controls
    _methods = ["(all)"] + sorted(status_df["method"].unique())
    _synths  = ["(all)"] + sorted(status_df["synth"].unique())
    _losses  = ["(all)"] + sorted(status_df["loss"].unique())

    method_filter = mo.ui.dropdown(options=_methods, value="(all)", label="Method")
    synth_filter  = mo.ui.dropdown(options=_synths,  value="(all)", label="Synth")
    loss_filter   = mo.ui.dropdown(options=_losses,  value="(all)", label="Loss")

    mo.hstack([method_filter, synth_filter, loss_filter])
    return loss_filter, method_filter, synth_filter


@app.cell
def _(loss_filter, method_filter, mo, status_df, synth_filter):
    _df = status_df.copy()
    if method_filter.value != "(all)":
        _df = _df[_df["method"] == method_filter.value]
    if synth_filter.value != "(all)":
        _df = _df[_df["synth"] == synth_filter.value]
    if loss_filter.value != "(all)":
        _df = _df[_df["loss"] == loss_filter.value]

    def _mark_n(v):
        if v == 0:   return f"✗ {v}"
        if v < 200:  return f"⚠ {v}"
        return f"✓ {v}"

    _display = _df.copy()
    _display["n trials"] = _display["n trials"].apply(_mark_n)

    mo.ui.table(_display, selection=None, label="Experiment status")
    return


@app.cell
def _(mo, status_df):
    # Pivot: synth × loss for each method — how many trials per cell
    _methods = sorted(status_df["method"].unique())
    _tabs = {}
    for _meth in _methods:
        _sub = status_df[status_df["method"] == _meth]
        _pivot = _sub.pivot_table(
            index="synth", columns="loss", values="n trials",
            aggfunc="sum", fill_value=0
        )
        def _mark(v):
            if v == 0:   return f"✗"
            if v < 200:  return f"⚠ {v}"
            return f"✓"
        _pivot = _pivot.map(_mark)
        _tabs[_meth] = mo.ui.table(_pivot.reset_index(), selection=None, label=_meth)

    mo.vstack([
        mo.md("## Coverage grid — trials per synth × loss cell"),
        mo.md("✓ = 200 trials complete  |  ⚠ = partial  |  ✗ = missing"),
        mo.tabs(_tabs),
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
