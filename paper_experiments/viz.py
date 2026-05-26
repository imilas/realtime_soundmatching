import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import pickle
    from pathlib import Path
    from itertools import combinations

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    RESULTS_DIR = Path("paper_experiments/results")
    METHOD_ORDER = ["GD", "HillClimber", "RandomSearch", "CMA-ES", "BO", "QL"]
    COLORS = {
        "GD": "black",
        "HillClimber": "#1f77b4",
        "RandomSearch": "#ff7f0e",
        "CMA-ES": "#2ca02c",
        "BO": "#d62728",
        "QL": "#9467bd",
    }
    return COLORS, METHOD_ORDER, RESULTS_DIR, mo, np, pd, pickle, plt


@app.cell
def _(RESULTS_DIR, pd, pickle):
    def _load_from_pkls(results_dir):
        _rows = []
        for _p in sorted(results_dir.glob("*.pkl")):
            _data = pickle.load(open(_p, "rb"))
            _rows.extend(_data["trials"])
        return pd.DataFrame(_rows)

    def _load_data(results_dir):
        _csv_path = results_dir / "results.csv"
        _pkls = list(results_dir.glob("*.pkl"))
        if not _pkls:
            return pd.DataFrame()
        _total = sum(len(pickle.load(open(_p, "rb"))["trials"]) for _p in _pkls)
        if _csv_path.exists():
            _df = pd.read_csv(_csv_path)
            if len(_df) == _total:
                return _df
        _df = _load_from_pkls(results_dir)
        _df.to_csv(_csv_path, index=False)
        return _df

    df = _load_data(RESULTS_DIR)
    synths_available = sorted(df["program"].unique()) if not df.empty else []
    methods_available = sorted(df["method"].unique()) if not df.empty else []
    print(f"Loaded {len(df)} trials | synths: {synths_available} | methods: {methods_available}")
    return df, methods_available, synths_available


@app.cell
def _(df):
    df
    return


@app.cell
def _(methods_available, mo, synths_available):
    synth_picker = mo.ui.multiselect(
        options=synths_available,
        value=synths_available,
        label="Synths",
    )
    method_picker = mo.ui.multiselect(
        options=methods_available,
        value=methods_available,
        label="Methods",
    )
    ql_tail_pct = mo.ui.slider(
        start=1,
        stop=100,
        step=1,
        value=100,
        label="QL last %",
    )
    mo.hstack([synth_picker, method_picker, ql_tail_pct])
    return method_picker, ql_tail_pct, synth_picker


@app.cell
def _(df, method_picker, np, pd, ql_tail_pct, synth_picker):
    def _keep_last_ql_pct(sub, pct):
        if sub.empty:
            return sub
        ql = sub[sub["method"] == "QL"]
        other = sub[sub["method"] != "QL"]
        if ql.empty:
            return sub

        kept = []
        for _synth, _rows in ql.groupby("program", sort=False):
            _n = max(1, int(np.ceil(len(_rows) * pct / 100.0)))
            kept.append(_rows.tail(_n))
        return pd.concat([other, *kept], ignore_index=True)

    selected = df[
        df["program"].isin(synth_picker.value) &
        df["method"].isin(method_picker.value)
    ]
    filtered = _keep_last_ql_pct(selected, ql_tail_pct.value)
    return (filtered,)


@app.cell
def _(COLORS, METHOD_ORDER, filtered, mo, plt):
    _synths = sorted(filtered["program"].unique())
    _methods = [_m for _m in METHOD_ORDER if _m in filtered["method"].unique()]

    if _synths and _methods:
        _fig, _axes = plt.subplots(1, len(_synths), figsize=(4 * len(_synths), 4), sharey=False)
        if len(_synths) == 1:
            _axes = [_axes]

        for _ax, _synth in zip(_axes, _synths):
            _data, _labels, _colors = [], [], []
            for _method in _methods:
                _vals = filtered[
                    (filtered["program"] == _synth) & (filtered["method"] == _method)
                ]["best_p_loss"].values
                if len(_vals):
                    _data.append(_vals)
                    _labels.append(_method)
                    _colors.append(COLORS.get(_method, "gray"))
            _bp = _ax.boxplot(_data, patch_artist=True, showfliers=True)
            for _patch, _color in zip(_bp["boxes"], _colors):
                _patch.set_facecolor(_color)
                _patch.set_alpha(0.6)
            _ax.set_xticks(range(1, len(_labels) + 1))
            _ax.set_xticklabels(_labels, rotation=35, ha="right", fontsize=8)
            _ax.set_title(_synth, fontsize=9)
            _ax.set_ylabel("Best P-Loss")
            _ax.grid(True, axis="y", alpha=0.3)

        _fig.suptitle("Best P-Loss Distribution per Method", fontsize=11)
        _fig.tight_layout()
        _out = mo.as_html(_fig)
        plt.close(_fig)
    else:
        _out = mo.md("_No data for current selection._")

    _out
    return


@app.cell
def _(COLORS, filtered, mo, np, plt):
    _ql = filtered[filtered["method"] == "QL"].copy()

    if _ql.empty:
        _out = mo.md("_No QL trials in current selection._")
    else:
        _synths = sorted(_ql["program"].unique())
        _fig, _axes = plt.subplots(1, len(_synths), figsize=(4 * len(_synths), 3.5), sharey=False)
        if len(_synths) == 1:
            _axes = [_axes]

        for _ax, _synth in zip(_axes, _synths):
            _vals = _ql[_ql["program"] == _synth]["best_p_loss"].values
            _x = np.arange(1, len(_vals) + 1)
            _w = max(1, len(_vals) // 5)
            _smooth = np.convolve(_vals, np.ones(_w) / _w, mode="valid")
            _sx = np.arange(_w, len(_vals) + 1)

            _ax.scatter(_x, _vals, color=COLORS["QL"], alpha=0.5, s=30, label="per-trial")
            _ax.plot(_sx, _smooth, color=COLORS["QL"], linewidth=2, label=f"window={_w}")
            _ax.set_xlabel("Trial #")
            _ax.set_ylabel("Best P-Loss")
            _ax.set_title(f"QL — {_synth}", fontsize=9)
            _ax.legend(fontsize=7)
            _ax.grid(True, alpha=0.3)

        _fig.suptitle("QL: P-Loss across trials (Q-table persists)", fontsize=10)
        _fig.tight_layout()
        _out = mo.as_html(_fig)
        plt.close(_fig)

    _out
    return


@app.cell
def _(METHOD_ORDER, filtered, mo, np, pd):
    if filtered.empty:
        mo.md("_No data._")
    else:
        _rows = []
        for _synth in sorted(filtered["program"].unique()):
            for _method in [_m for _m in METHOD_ORDER if _m in filtered["method"].unique()]:
                _vals = filtered[
                    (filtered["program"] == _synth) & (filtered["method"] == _method)
                ]["best_p_loss"].values
                if len(_vals):
                    _rows.append({
                        "synth": _synth,
                        "method": _method,
                        "n": len(_vals),
                        "mean": round(float(np.mean(_vals)), 4),
                        "std": round(float(np.std(_vals)), 4),
                        "min": round(float(np.min(_vals)), 4),
                    })
        mo.ui.table(pd.DataFrame(_rows), selection=None)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
