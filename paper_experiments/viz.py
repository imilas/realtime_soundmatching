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
    return COLORS, METHOD_ORDER, RESULTS_DIR, combinations, mo, np, pd, pickle, plt, Path


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
    mo.hstack([synth_picker, method_picker])
    return method_picker, synth_picker


@app.cell
def _(df, method_picker, synth_picker):
    filtered = df[
        df["program"].isin(synth_picker.value) &
        df["method"].isin(method_picker.value)
    ]
    return (filtered,)


# ---------------------------------------------------------------------------
# Boxplots
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# NPSK win-rate heatmaps
# ---------------------------------------------------------------------------
@app.cell
def _(METHOD_ORDER, filtered, mo, np, plt):
    def _npsk(sub, methods):
        _mat = np.full((len(methods), len(methods)), np.nan)
        for _i, _a in enumerate(methods):
            for _j, _b in enumerate(methods):
                if _i == _j:
                    _mat[_i, _j] = 0.5
                    continue
                _va = sub[sub["method"] == _a]["best_p_loss"].values
                _vb = sub[sub["method"] == _b]["best_p_loss"].values
                _n = min(len(_va), len(_vb))
                if _n:
                    _mat[_i, _j] = (_va[:_n] < _vb[:_n]).mean()
        return _mat

    _synths = sorted(filtered["program"].unique())
    _methods = [_m for _m in METHOD_ORDER if _m in filtered["method"].unique()]

    if _synths and len(_methods) >= 2:
        _fig, _axes = plt.subplots(1, len(_synths), figsize=(3.5 * len(_synths), 3.5))
        if len(_synths) == 1:
            _axes = [_axes]

        for _ax, _synth in zip(_axes, _synths):
            _mat = _npsk(filtered[filtered["program"] == _synth], _methods)
            _im = _ax.imshow(_mat, vmin=0, vmax=1, cmap="RdYlGn")
            _ax.set_xticks(range(len(_methods)))
            _ax.set_xticklabels(_methods, rotation=45, ha="right", fontsize=7)
            _ax.set_yticks(range(len(_methods)))
            _ax.set_yticklabels(_methods, fontsize=7)
            _ax.set_title(_synth, fontsize=8)
            for _i in range(len(_methods)):
                for _j in range(len(_methods)):
                    _v = _mat[_i, _j]
                    if not np.isnan(_v):
                        _ax.text(_j, _i, f"{_v:.2f}", ha="center", va="center", fontsize=7)
            plt.colorbar(_im, ax=_ax, fraction=0.046)

        _fig.suptitle("NPSK win rate (row beats column)", fontsize=10)
        _fig.tight_layout()
        _out = mo.as_html(_fig)
        plt.close(_fig)
    else:
        _out = mo.md("_Need at least 2 methods selected._")

    _out


# ---------------------------------------------------------------------------
# QL cross-trial learning
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
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


if __name__ == "__main__":
    app.run()
