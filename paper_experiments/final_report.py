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
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import io

    from paper_experiments.make_verification_report import (
        METHODS,
        LOSSES,
        LOSSES_ALL,
        SYNTHS,
        SYNTH_LABELS,
        PUBLISHED,
        load_pkl,
        load_trials,
        extract_scores,
        bootstrap_medians,
        npsk_ranks,
        discover_methods,
    )

    RES = Path(__file__).parent / "results"
    FIG = RES / "figures"
    return (
        FIG,
        LOSSES_ALL,
        PUBLISHED,
        SYNTHS,
        SYNTH_LABELS,
        bootstrap_medians,
        cm,
        discover_methods,
        extract_scores,
        io,
        load_trials,
        mo,
        np,
        npsk_ranks,
        pd,
        plt,
    )


@app.cell
def _(SYNTHS, SYNTH_LABELS, discover_methods, mo):
    all_methods = discover_methods()
    synth_select = mo.ui.multiselect(
        options={SYNTH_LABELS[s]: s for s in SYNTHS},
        value=[SYNTH_LABELS[s] for s in SYNTHS],
        label="Synths to show",
    )
    method_select = mo.ui.multiselect(
        options=all_methods,
        value=all_methods,
        label="Methods to compare",
    )
    mo.vstack(
        [
            mo.md("### Select synths and methods"),
            synth_select,
            method_select,
        ]
    )
    return all_methods, method_select, synth_select


@app.cell
def _(LOSSES_ALL, SYNTHS, all_methods, load_trials):
    trial_cache = {
        (s, l, m): load_trials(s, l, m)
        for s in SYNTHS
        for l in LOSSES_ALL
        for m in all_methods
    }
    return (trial_cache,)


@app.cell
def _(all_methods, cm, np):
    _gd_sorted_mc = sorted(
        [m for m in all_methods if m == "GD" or m.startswith("GD_lr")],
        key=lambda m: float(m[5:]) if m.startswith("GD_lr") else 0.045,
    )
    _reds_mc = cm.Reds(np.linspace(0.35, 0.9, max(len(_gd_sorted_mc), 1)))
    _BASE_MC = {"CMA-ES": "#3cb44b", "LES": "#4363d8", "RandomSearch": "#f58231"}


    def method_color(method):
        if method in _BASE_MC:
            return _BASE_MC[method]
        idx = _gd_sorted_mc.index(method) if method in _gd_sorted_mc else 0
        r, g, b, _ = _reds_mc[idx]
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    return (method_color,)


@app.cell
def _(
    LOSSES_ALL,
    PUBLISHED,
    SYNTHS,
    SYNTH_LABELS,
    bootstrap_medians,
    extract_scores,
    method_select,
    mo,
    npsk_ranks,
    pd,
    synth_select,
    trial_cache,
):
    def _rank_or_dash(r):
        return r if isinstance(r, int) else "—"


    _selected = [s for s in SYNTHS if s in synth_select.value] or list(SYNTHS)
    _methods = method_select.value or ["GD"]
    _tables = []
    for _synth in _selected:
        _rows = []
        _pub_row = {"": "Published"}
        for _loss in LOSSES_ALL:
            _pub_row[_loss] = PUBLISHED[_synth].get(_loss, "—")
        _rows.append(_pub_row)

        for _method in _methods:
            _groups = {}
            for _loss in LOSSES_ALL:
                _trials = trial_cache.get((_synth, _loss, _method), [])
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
            mo.vstack(
                [
                    mo.md(f"### {SYNTH_LABELS[_synth]} (`{_synth}`)"),
                    mo.ui.table(
                        _df.reset_index(),
                        selection=None,
                        label=SYNTH_LABELS[_synth],
                    ),
                ]
            )
        )

    mo.vstack(_tables)
    return


@app.cell
def _(
    LOSSES_ALL,
    SYNTHS,
    SYNTH_LABELS,
    io,
    method_color,
    method_select,
    mo,
    np,
    plt,
    synth_select,
    trial_cache,
):
    plt.rcParams.update({"font.size": 30})
    _selected = [s for s in SYNTHS if s in synth_select.value] or list(SYNTHS)
    _methods = method_select.value or []
    _n_rows = len(LOSSES_ALL)
    _n_cols = len(_selected)

    _fig, _axes = plt.subplots(
        _n_rows,
        _n_cols,
        figsize=(8.0 * _n_cols, 6.4 * _n_rows),
        squeeze=False,
    )

    for _ri, _loss in enumerate(LOSSES_ALL):
        for _ci, _synth in enumerate(_selected):
            _ax = _axes[_ri][_ci]
            _any = False

            for _method in _methods:
                _trials = trial_cache.get((_synth, _loss, _method), [])
                if not _trials:
                    continue

                _B = max(
                    (len(t.get("history_p_loss", [])) for t in _trials),
                    default=0,
                )
                if _B == 0:
                    continue

                _curves = []

                for _t in _trials:
                    _c = np.fmin.accumulate(
                        np.asarray(_t.get("history_p_loss", []), dtype=float)
                    )

                    if len(_c) < _B:
                        _c = np.pad(
                            _c,
                            (0, _B - len(_c)),
                            constant_values=_c[-1] if len(_c) else np.nan,
                        )

                    _curves.append(_c)

                _arr = np.vstack(_curves)
                _x = np.arange(1, _B + 1)
                _med = np.nanmedian(_arr, 0)

                _ax.plot(
                    _x,
                    _med,
                    color=method_color(_method),
                    lw=1.6,
                    label=_method.replace("GD_lr", "lr="),
                )

                _any = True

            if _ri == _n_rows - 1:
                _ax.set_xlabel("evaluations", fontsize=15)

            if _ci == 0:
                _ax.set_ylabel(f"{_loss}\nbest-so-far P-loss (log)", fontsize=30)

            if _ri == 0:
                _ax.set_title(SYNTH_LABELS.get(_synth, _synth), fontsize=30)

            _ax.grid(True, alpha=0.3, which="both")

            if _any:
                _ax.set_yscale("log")

            if _ri == 0 and _ci == 0 and _any:
                _ax.legend(fontsize=15)

    _fig.suptitle(
        "Sample efficiency: median best-so-far P-loss, rows=loss, cols=synth",
        fontsize=30,
    )

    _fig.tight_layout()

    _buf = io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(_fig)

    _buf.seek(0)
    mo.Html(f'<div style="overflow-x:auto"><img src="data:image/png;base64,{__import__("base64").b64encode(_buf.read()).decode()}"style="min-width:2000px"></div>')
    return


@app.cell
def _(
    LOSSES_ALL,
    SYNTHS,
    SYNTH_LABELS,
    io,
    method_color,
    method_select,
    mo,
    np,
    plt,
    synth_select,
    trial_cache,
):
    plt.rcParams.update({"font.size": 30})
    _selected = [s for s in SYNTHS if s in synth_select.value] or list(SYNTHS)
    _methods = method_select.value or []
    _n_rows = len(LOSSES_ALL)
    _n_cols = len(_selected)

    _fig, _axes = plt.subplots(
        _n_rows, _n_cols, figsize=(8.0 * _n_cols, 6.4 * _n_rows), squeeze=False
    )

    for _ri, _loss in enumerate(LOSSES_ALL):
        for _ci, _synth in enumerate(_selected):
            _ax = _axes[_ri][_ci]
            _any = False
            for _method in _methods:
                _trials = trial_cache.get((_synth, _loss, _method), [])
                if not _trials:
                    continue
                _B = max(
                    (len(t.get("history_p_loss", [])) for t in _trials), default=0
                )
                if _B == 0:
                    continue
                _curves = []
                for _t in _trials:
                    _c = np.asarray(_t.get("history_p_loss", []), dtype=float)
                    if len(_c) < _B:
                        _c = np.pad(_c, (0, _B - len(_c)), constant_values=np.nan)
                    _curves.append(_c)
                _arr = np.vstack(_curves)
                _x = np.arange(1, _B + 1)
                _med = np.nanmedian(_arr, 0)
                _ax.plot(
                    _x,
                    _med,
                    color=method_color(_method),
                    lw=1.6,
                    label=_method.replace("GD_lr", "lr="),
                )
                _any = True
            if _ri == _n_rows - 1:
                _ax.set_xlabel("evaluations", fontsize=15)
            if _ci == 0:
                _ax.set_ylabel(f"{_loss}\nreturned P-loss", fontsize=30)
            if _ri == 0:
                _ax.set_title(SYNTH_LABELS.get(_synth, _synth), fontsize=30)
            _ax.grid(True, alpha=0.3, which="both")
            if _ri == 0 and _ci == 0 and _any:
                _ax.legend(fontsize=15)

    _fig.suptitle(
        "Sanity check: median returned (instantaneous) P-loss, rows=loss, cols=synth",
        fontsize=30,
    )
    _fig.tight_layout()
    _buf = io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(_fig)
    _buf.seek(0)
    mo.Html(f'<div style="overflow-x:auto"><img src="data:image/png;base64,{__import__("base64").b64encode(_buf.read()).decode()}"style="min-width:2000px"></div>')
    return


@app.cell
def _(
    LOSSES_ALL,
    SYNTHS,
    SYNTH_LABELS,
    io,
    method_color,
    method_select,
    mo,
    np,
    plt,
    synth_select,
    trial_cache,
):
    plt.rcParams.update({"font.size": 30})
    _selected = [s for s in SYNTHS if s in synth_select.value] or list(SYNTHS)
    _methods = method_select.value or []
    _n_rows = len(LOSSES_ALL)
    _n_cols = len(_selected)

    _fig, _axes = plt.subplots(
        _n_rows, _n_cols, figsize=(8.0 * _n_cols, 6.4 * _n_rows), squeeze=False
    )

    for _ri, _loss in enumerate(LOSSES_ALL):
        for _ci, _synth in enumerate(_selected):
            _ax = _axes[_ri][_ci]
            _any = False
            for _method in _methods:
                _trials = trial_cache.get((_synth, _loss, _method), [])
                if not _trials:
                    continue
                _curves = [
                    np.asarray(t.get("history_audio_loss", []), dtype=float)
                    for t in _trials
                ]
                _curves = [c for c in _curves if len(c) > 0]
                if not _curves:
                    continue
                _B = min(len(c) for c in _curves)
                _arr = np.stack([c[:_B] for c in _curves])
                _x = np.arange(1, _B + 1)
                _med = np.nanmedian(_arr, 0)
                _ax.plot(
                    _x,
                    _med,
                    color=method_color(_method),
                    lw=1.6,
                    label=_method.replace("GD_lr", "lr="),
                )
                _any = True
            if _ri == _n_rows - 1:
                _ax.set_xlabel("evaluations", fontsize=15)
            if _ci == 0:
                _ax.set_ylabel(f"{_loss}\naudio loss", fontsize=30)
            if _ri == 0:
                _ax.set_title(SYNTH_LABELS.get(_synth, _synth), fontsize=30)
            _ax.grid(True, alpha=0.3, which="both")
            if _any:
                _ax.set_yscale("symlog", linthresh=1e-6)
            if _ri == 0 and _ci == 0 and _any:            
                _ax.legend(fontsize=15)

    _fig.suptitle(
        "Median audio-loss curves, rows=loss, cols=synth", fontsize=30
    )
    _fig.tight_layout()
    _buf = io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(_fig)


    _buf.seek(0)
    mo.Html(f'<div style="overflow-x:auto"><img src="data:image/png;base64,{__import__("base64").b64encode(_buf.read()).decode()}"style="min-width:2000px"></div>')
    # _buf.seek(0)
    # mo.image(_buf.read())
    return


@app.cell
def _(FIG, mo):
    mo.image(str(FIG / "06_identifiability_scatter.png"))
    return


if __name__ == "__main__":
    app.run()
