import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import os
    import pickle
    import re
    import sys
    from pathlib import Path

    os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "mpl"))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from paper_experiments.config import METHODS, SYNTH_LOSS, SYNTHS
    from utils.loss_functions import ALL_LOSSES

    return ALL_LOSSES, Path, SYNTHS, SYNTH_LOSS, mo, np, pd, pickle, plt, re


@app.cell(hide_code=True)
def _(Path):
    results_dir = Path(__file__).parent.parent / "paper_experiments" / "results"
    method_names = ["GD", "RandomSearch", "CMA-ES", "BO"]
    metric_names = [
        "median_returned_p_loss",
        "median_visited_p_loss",
        "median_deception_gap",
        "median_best_audio_loss",
        "median_duration_s",
    ]
    method_colors = {
        "GD": "black",
        "RandomSearch": "#ff7f0e",
        "CMA-ES": "#2ca02c",
        "BO": "#d62728",
    }
    return method_colors, method_names, metric_names, results_dir


@app.cell(hide_code=True)
def _(ALL_LOSSES):
    loss_names = list(ALL_LOSSES)
    return (loss_names,)


@app.cell(hide_code=True)
def _(mo):
    _out = mo.md("""
    # Multi-Loss Results

    This notebook compares every completed `(synth, loss, method)` result cell
    found in `paper_experiments/results/`. It supports partial grids, so you can
    open it while experiments are still running and refresh after more cells finish.
    """)
    _out
    return


@app.cell(hide_code=True)
def _(SYNTH_LOSS, np, pickle, re, results_dir):
    def slug_loss_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


    def result_paths(synth: str, loss_name: str, method: str):
        _explicit_path = results_dir / f"{synth}_{slug_loss_name(loss_name)}_{method}.pkl"
        _paths = [(_explicit_path, "explicit")]
        if loss_name == SYNTH_LOSS.get(synth):
            _paths.append((results_dir / f"{synth}_{method}.pkl", "default"))
        return _paths


    def load_result_trials(synth: str, loss_name: str, method: str):
        for _path, _source in result_paths(synth, loss_name, method):
            if _path.exists():
                with _path.open("rb") as _handle:
                    _data = pickle.load(_handle)
                return _data.get("trials", []), _path, _source
        return [], None, None


    def returned_p_loss(trial: dict) -> float:
        _audio_loss = np.asarray(trial.get("history_audio_loss", []), dtype=float)
        _p_loss = np.asarray(trial.get("history_p_loss", []), dtype=float)
        if len(_audio_loss) == 0 or len(_p_loss) == 0:
            return float("nan")
        return float(_p_loss[int(np.nanargmin(_audio_loss))])


    def visited_p_loss(trial: dict) -> float:
        _p_loss = np.asarray(trial.get("history_p_loss", []), dtype=float)
        if len(_p_loss) == 0:
            return float("nan")
        return float(np.nanmin(_p_loss))


    def best_audio_loss(trial: dict) -> float:
        _audio_loss = np.asarray(trial.get("history_audio_loss", []), dtype=float)
        if len(_audio_loss) == 0:
            return float("nan")
        return float(np.nanmin(_audio_loss))

    return best_audio_loss, load_result_trials, returned_p_loss, visited_p_loss


@app.cell(hide_code=True)
def _(
    SYNTHS,
    best_audio_loss,
    load_result_trials,
    loss_names,
    method_names,
    np,
    pd,
    returned_p_loss,
    visited_p_loss,
):
    _trial_rows = []
    _history_rows = []
    for _synth in SYNTHS:
        for _loss_name in loss_names:
            for _method in method_names:
                _trials, _path, _source = load_result_trials(_synth, _loss_name, _method)
                if not _trials:
                    continue
                for _trial_idx, _trial in enumerate(_trials):
                    _returned = returned_p_loss(_trial)
                    _visited = visited_p_loss(_trial)
                    _trial_rows.append(
                        {
                            "synth": _synth,
                            "loss": _loss_name,
                            "method": _method,
                            "trial": _trial_idx,
                            "returned_p_loss": _returned,
                            "visited_p_loss": _visited,
                            "deception_gap": _returned - _visited,
                            "saved_best_p_loss": float(_trial.get("best_p_loss", np.nan)),
                            "best_audio_loss": best_audio_loss(_trial),
                            "eval_budget": int(_trial.get("eval_budget", 0) or 0),
                            "duration_s": float(_trial.get("duration_s", np.nan)),
                            "source": _source,
                            "file": _path.name,
                        }
                    )
                    _best_so_far = np.minimum.accumulate(
                        np.asarray(_trial.get("history_p_loss", []), dtype=float)
                    )
                    for _eval_idx, _value in enumerate(_best_so_far, start=1):
                        _history_rows.append(
                            {
                                "synth": _synth,
                                "loss": _loss_name,
                                "method": _method,
                                "trial": _trial_idx,
                                "eval": _eval_idx,
                                "best_so_far_p_loss": float(_value),
                            }
                        )

    trials_df = pd.DataFrame(_trial_rows)
    history_df = pd.DataFrame(_history_rows)
    return history_df, trials_df


@app.cell(hide_code=True)
def _(pd, trials_df):
    if len(trials_df):
        summary_df = (
            trials_df.groupby(["synth", "loss", "method"], as_index=False)
            .agg(
                n=("trial", "count"),
                median_returned_p_loss=("returned_p_loss", "median"),
                median_visited_p_loss=("visited_p_loss", "median"),
                median_deception_gap=("deception_gap", "median"),
                median_saved_best_p_loss=("saved_best_p_loss", "median"),
                median_best_audio_loss=("best_audio_loss", "median"),
                median_duration_s=("duration_s", "median"),
                eval_budget=("eval_budget", "max"),
                source=("source", lambda _values: ",".join(sorted(set(_values)))),
            )
            .sort_values(["synth", "loss", "median_returned_p_loss", "method"])
            .reset_index(drop=True)
        )
    else:
        summary_df = pd.DataFrame()
    return (summary_df,)


@app.cell(hide_code=True)
def _(mo, summary_df, trials_df):
    if len(summary_df) == 0:
        _out = mo.md("No multi-loss result files found yet in `paper_experiments/results/`.")
    else:
        _out = mo.md(
            f"""
            **Loaded `{len(trials_df):,}` trials across `{len(summary_df):,}` completed cells.**

            Main metrics:

            - `returned_p_loss`: P-loss at the candidate selected by minimum audio loss.
            - `visited_p_loss`: oracle best P-loss visited during the run.
            - `deception_gap`: `returned_p_loss - visited_p_loss`; larger means the loss selected a worse parameter set than the method had visited.
            """
        )
    _out
    return


@app.cell(hide_code=True)
def _(loss_names, metric_names, mo, summary_df):
    loss_options = (
        list(summary_df["loss"].drop_duplicates())
        if len(summary_df)
        else loss_names
    )
    selected_loss = mo.ui.dropdown(
        options=loss_options,
        value=loss_options[0] if loss_options else None,
        label="Loss",
    )
    selected_metric = mo.ui.dropdown(
        options=metric_names,
        value="median_returned_p_loss",
        label="Metric",
    )
    _out = mo.hstack([selected_loss, selected_metric])
    _out
    return selected_loss, selected_metric


@app.cell(hide_code=True)
def _(selected_loss, summary_df):
    if selected_loss.value is None:
        selected_loss_summary_df = summary_df.iloc[0:0].copy()
    else:
        selected_loss_summary_df = summary_df[
            summary_df["loss"] == selected_loss.value
        ].sort_values(["synth", "median_returned_p_loss", "method"]).copy()
    return (selected_loss_summary_df,)


@app.cell(hide_code=True)
def _(pd, selected_loss_summary_df, selected_metric):
    if len(selected_loss_summary_df):
        loss_leaderboard_df = (
            selected_loss_summary_df.groupby("method", as_index=False)
            .agg(
                synths=("synth", "nunique"),
                cells=("synth", "count"),
                trials=("n", "sum"),
                median_metric=(selected_metric.value, "median"),
                mean_metric=(selected_metric.value, "mean"),
                worst_metric=(selected_metric.value, "max"),
            )
            .sort_values(["median_metric", "mean_metric", "method"])
            .reset_index(drop=True)
        )
        loss_leaderboard_df.insert(0, "rank", range(1, len(loss_leaderboard_df) + 1))
    else:
        loss_leaderboard_df = pd.DataFrame()
    return (loss_leaderboard_df,)


@app.cell(hide_code=True)
def _(
    loss_leaderboard_df,
    mo,
    selected_loss,
    selected_loss_summary_df,
    selected_metric,
):
    if selected_loss.value is None or len(selected_loss_summary_df) == 0:
        _out = mo.callout("No completed cells for the selected loss.", kind="warn")
    else:
        _best = loss_leaderboard_df.iloc[0] if len(loss_leaderboard_df) else None
        _n_trials = int(selected_loss_summary_df["n"].sum())
        _n_synths = int(selected_loss_summary_df["synth"].nunique())
        _n_methods = int(selected_loss_summary_df["method"].nunique())
        _best_text = "n/a" if _best is None else f"{_best['method']} ({_best['median_metric']:.4g})"
        _out = mo.vstack([
            mo.md(f"## Browse results for `{selected_loss.value}`"),
            mo.hstack([
                mo.stat(_best_text, label="Best method", caption=f"by {selected_metric.value}", bordered=True),
                mo.stat(_n_synths, label="Synths", caption="with completed cells", bordered=True),
                mo.stat(_n_methods, label="Methods", caption="with completed cells", bordered=True),
                mo.stat(f"{_n_trials:,}", label="Trials", caption="loaded for this loss", bordered=True),
            ]),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(loss_leaderboard_df, mo, selected_loss_summary_df):
    formatted_loss_summary_df = selected_loss_summary_df.copy()
    formatted_loss_leaderboard_df = loss_leaderboard_df.copy()
    _numeric_cols = [
        "median_returned_p_loss",
        "median_visited_p_loss",
        "median_deception_gap",
        "median_saved_best_p_loss",
        "median_best_audio_loss",
        "median_duration_s",
    ]
    for _col in _numeric_cols:
        if _col in formatted_loss_summary_df:
            formatted_loss_summary_df[_col] = formatted_loss_summary_df[_col].map(lambda _value: f"{_value:.4g}")
    for _col in ["median_metric", "mean_metric", "worst_metric"]:
        if _col in formatted_loss_leaderboard_df:
            formatted_loss_leaderboard_df[_col] = formatted_loss_leaderboard_df[_col].map(lambda _value: f"{_value:.4g}")
    _out = mo.tabs({
        "Leaderboard": mo.ui.table(
            formatted_loss_leaderboard_df,
            label="Method leaderboard across synthesizers",
            page_size=10,
        ),
        "Cell details": mo.ui.table(
            formatted_loss_summary_df,
            label="Completed synth/method cells",
            page_size=20,
        ),
    })
    _out
    return


@app.cell(hide_code=True)
def _(
    method_names,
    mo,
    np,
    plt,
    selected_loss,
    selected_loss_summary_df,
    selected_metric,
):
    if selected_loss.value is None or len(selected_loss_summary_df) == 0:
        _out = mo.md("No completed cells for the selected loss.")
    else:
        _section = mo.md("### Synth × Method Heatmap")
        _pivot = selected_loss_summary_df.pivot_table(
            index="synth",
            columns="method",
            values=selected_metric.value,
            aggfunc="median",
        )
        _pivot = _pivot[[_method for _method in method_names if _method in _pivot.columns]]
        _fig, _ax = plt.subplots(figsize=(10, max(3.8, 0.55 * len(_pivot))))
        _data = _pivot.to_numpy(dtype=float)
        _masked_data = np.ma.masked_invalid(_data)
        _image = _ax.imshow(_masked_data, aspect="auto", cmap="viridis_r")
        _ax.set_xticks(np.arange(len(_pivot.columns)))
        _ax.set_xticklabels(_pivot.columns, rotation=35, ha="right")
        _ax.set_yticks(np.arange(len(_pivot.index)))
        _ax.set_yticklabels(_pivot.index)
        _ax.set_title(f"{selected_loss.value}: {selected_metric.value}")
        for _y in range(_data.shape[0]):
            for _x in range(_data.shape[1]):
                if np.isfinite(_data[_y, _x]):
                    _ax.text(
                        _x,
                        _y,
                        f"{_data[_y, _x]:.3g}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )
        _fig.colorbar(_image, ax=_ax, label=selected_metric.value)
        _fig.tight_layout()
        _out = mo.vstack([_section, mo.as_html(_fig)])
        plt.close(_fig)
    _out
    return


@app.cell(hide_code=True)
def _(
    method_colors,
    mo,
    np,
    plt,
    selected_loss,
    selected_loss_summary_df,
    selected_metric,
):
    if selected_loss.value is None or len(selected_loss_summary_df) == 0:
        _out = mo.md("No bar chart data for the selected loss.")
    else:
        _section = mo.md("### Grouped Method Comparison")
        _synths = list(selected_loss_summary_df["synth"].drop_duplicates())
        _methods = [
            _method for _method in method_colors
            if _method in set(selected_loss_summary_df["method"])
        ]
        _x_values = np.arange(len(_synths))
        _width = 0.82 / max(len(_methods), 1)
        _fig, _ax = plt.subplots(figsize=(10, 4.8))
        for _idx, _method in enumerate(_methods):
            _method_df = selected_loss_summary_df[
                selected_loss_summary_df["method"] == _method
            ].set_index("synth")
            _values = [
                _method_df.loc[_synth, selected_metric.value]
                if _synth in _method_df.index
                else np.nan
                for _synth in _synths
            ]
            _ax.bar(
                _x_values + (_idx - (len(_methods) - 1) / 2) * _width,
                _values,
                _width,
                label=_method,
                color=method_colors.get(_method, "#777777"),
                alpha=0.85,
            )
        _ax.set_xticks(_x_values)
        _ax.set_xticklabels(_synths, rotation=20, ha="right")
        _ax.set_ylabel(selected_metric.value)
        _ax.set_title(f"{selected_loss.value}: synth comparison by method")
        _ax.grid(True, axis="y", alpha=0.25)
        _ax.legend(fontsize=8)
        _fig.tight_layout()
        _out = mo.vstack([_section, mo.as_html(_fig)])
        plt.close(_fig)
    _out
    return


@app.cell(hide_code=True)
def _(SYNTHS, mo, selected_loss_summary_df):
    synth_options = (
        list(selected_loss_summary_df["synth"].drop_duplicates())
        if len(selected_loss_summary_df)
        else list(SYNTHS)
    )
    selected_synth = mo.ui.dropdown(
        options=synth_options,
        value=synth_options[0] if synth_options else None,
        label="Synth drilldown",
    )
    _out = mo.vstack([
        mo.md("## Drill Down"),
        mo.hstack([selected_synth]),
    ])
    _out
    return (selected_synth,)


@app.cell(hide_code=True)
def _(
    method_colors,
    mo,
    np,
    plt,
    selected_loss,
    selected_loss_summary_df,
    selected_metric,
    selected_synth,
):
    if selected_loss.value is None:
        _out = mo.md("No loss selected.")
    else:
        _cell_summary_df = selected_loss_summary_df[
            selected_loss_summary_df["synth"] == selected_synth.value
        ].sort_values(selected_metric.value)
        if len(_cell_summary_df) == 0:
            _out = mo.md("No completed method cells for this synth/loss.")
        else:
            _fig, _ax = plt.subplots(figsize=(8.5, 4.2))
            _methods = list(_cell_summary_df["method"])
            _values = _cell_summary_df[selected_metric.value].to_numpy(dtype=float)
            _bars = _ax.bar(
                np.arange(len(_methods)),
                _values,
                color=[method_colors.get(_method, "#777777") for _method in _methods],
                alpha=0.85,
            )
            _ax.set_xticks(np.arange(len(_methods)))
            _ax.set_xticklabels(_methods, rotation=25, ha="right")
            _ax.set_ylabel(selected_metric.value)
            _ax.set_title(f"{selected_synth.value} | {selected_loss.value}")
            _ax.grid(True, axis="y", alpha=0.25)
            for _bar, _value in zip(_bars, _values):
                _ax.text(
                    _bar.get_x() + _bar.get_width() / 2,
                    _bar.get_height(),
                    f"{_value:.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            _fig.tight_layout()
            _out = mo.as_html(_fig)
            plt.close(_fig)
    _out
    return


@app.cell(hide_code=True)
def _(history_df, selected_loss, selected_synth):
    if selected_loss.value is None or len(history_df) == 0:
        selected_history_df = history_df.iloc[0:0].copy()
    else:
        selected_history_df = history_df[
            (history_df["synth"] == selected_synth.value)
            & (history_df["loss"] == selected_loss.value)
        ].copy()
    return (selected_history_df,)


@app.cell(hide_code=True)
def _(
    method_colors,
    mo,
    np,
    plt,
    selected_history_df,
    selected_loss,
    selected_synth,
):
    if selected_loss.value is None or len(selected_history_df) == 0:
        _out = mo.md("No learning-curve data available.")
    else:
        _fig, _ax = plt.subplots(figsize=(9, 4.8))
        for _method, _method_history_df in selected_history_df.groupby("method"):
            _curves = []
            _max_eval = int(_method_history_df["eval"].max())
            for _, _trial_history_df in _method_history_df.groupby("trial"):
                _trial_curve = (
                    _trial_history_df.sort_values("eval")["best_so_far_p_loss"]
                    .to_numpy(dtype=float)
                )
                if len(_trial_curve) < _max_eval:
                    _trial_curve = np.pad(
                        _trial_curve,
                        (0, _max_eval - len(_trial_curve)),
                        constant_values=_trial_curve[-1],
                    )
                _curves.append(_trial_curve)
            _curve_array = np.vstack(_curves)
            _x_values = np.arange(1, _max_eval + 1)
            _median = np.nanmedian(_curve_array, axis=0)
            _q25 = np.nanpercentile(_curve_array, 25, axis=0)
            _q75 = np.nanpercentile(_curve_array, 75, axis=0)
            _color = method_colors.get(_method, "#777777")
            _ax.plot(
                _x_values,
                _median,
                label=f"{_method} (n={len(_curves)})",
                color=_color,
                linewidth=1.8,
            )
            _ax.fill_between(_x_values, _q25, _q75, color=_color, alpha=0.12)
        _ax.set_yscale("log")
        _ax.set_xlabel("Evaluation")
        _ax.set_ylabel("Best-so-far P-loss")
        _ax.set_title(f"Learning curves: {selected_synth.value} | {selected_loss.value}")
        _ax.grid(True, which="both", alpha=0.25)
        _ax.legend(fontsize=8)
        _fig.tight_layout()
        _out = mo.as_html(_fig)
        plt.close(_fig)
    _out
    return


@app.cell(hide_code=True)
def _(mo, selected_loss, selected_synth, summary_df):
    if selected_loss.value is None:
        _out = mo.md("")
    else:
        _ranked_df = summary_df[
            (summary_df["synth"] == selected_synth.value)
            & (summary_df["loss"] == selected_loss.value)
        ].sort_values("median_returned_p_loss")
        if len(_ranked_df) == 0:
            _out = mo.md("")
        else:
            _best_row = _ranked_df.iloc[0]
            _out = mo.md(
                f"""
                **Best returned P-loss for this cell:** `{_best_row['method']}` with
                median returned P-loss `{_best_row['median_returned_p_loss']:.4g}`
                across `{int(_best_row['n'])}` trials.
                """
            )
    _out
    return


if __name__ == "__main__":
    app.run()
