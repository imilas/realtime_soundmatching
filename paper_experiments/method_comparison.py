import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import re
    import pickle
    import marimo as mo
    import numpy as np
    import pandas as pd
    from scipy.stats import mannwhitneyu

    RES = Path(__file__).parent / "results"

    METHODS = ["GD", "CMA-ES", "RandomSearch", "LES"]
    LOSSES = ["SIMSE_Spec", "L1_Spec", "JTFS", "DTW_Envelope", "CLAP"]

    # Synths with both replication data and a published P-Loss NPSK rank to
    # compare against (matches gd_verification.py / make_verification_report.py).
    SYNTHS = [
        "bandpass_noise",
        "add_sinesaw",
        "am_noise",
        "sine_mod_saw",
        "chirplet",
    ]

    # Published P-Loss NPSK ranks. Rank 1 = best (lowest expected P-Loss).
    PUBLISHED_RANKS = {
        # IEEE 2025 in-domain paper, Table III
        "bandpass_noise": {"SIMSE_Spec": 1, "L1_Spec": 1, "JTFS": 4, "DTW_Envelope": 3},
        "add_sinesaw":       {"SIMSE_Spec": 4, "L1_Spec": 2, "JTFS": 1, "DTW_Envelope": 3},
        "am_noise":          {"SIMSE_Spec": 4, "L1_Spec": 2, "JTFS": 3, "DTW_Envelope": 1},
        "sine_mod_saw":      {"SIMSE_Spec": 2, "L1_Spec": 3, "JTFS": 4, "DTW_Envelope": 1},
        # ISMIR paper, Table 1 (in-domain, chirplet only)
        "chirplet":          {"SIMSE_Spec": 3, "L1_Spec": 1, "JTFS": 1, "DTW_Envelope": 4},
    }
    return (
        LOSSES,
        METHODS,
        PUBLISHED_RANKS,
        RES,
        SYNTHS,
        mannwhitneyu,
        mo,
        np,
        pd,
        pickle,
        re,
    )


@app.cell
def _(mo):
    mo.md("""
    # Iterative-method verification & comparison

    Replication of the IEEE 2025 / ISMIR (OOD) in-domain papers' P-Loss NPSK
    rankings, run with **all search methods** (GD, CMA-ES,
    RandomSearch) instead of just GD — plus a cross-method comparison in
    the spirit of `findings.py`.

    Pick a **method** below to see its data availability and how its
    computed ranks compare to the published ones (mirrors
    `verification_report.html`, generalized to every method). Further down,
    pick a **synth × loss** cell to compare all methods head-to-head.
    """)
    return


@app.cell
def _(RES, np, pickle, re):
    def _returned_p_loss(trial, method):
        """Final P-loss for GD (its trajectory is the optimization itself);
        P-loss at the best-audio-loss step for black-box methods (their
        'returned' candidate is the best one found, not the last sampled)."""
        ploss = np.asarray(trial.get("history_p_loss", []), dtype=float)
        if len(ploss) == 0:
            return float("nan")
        if method == "GD":
            return float(ploss[-1])
        audio = np.asarray(trial.get("history_audio_loss", []), dtype=float)
        return float(ploss[int(np.nanargmin(audio))]) if len(audio) else float("nan")

    def _slug(loss):
        return re.sub(r"[^A-Za-z0-9]+", "_", loss).strip("_")

    def load_trials(synth, loss, method):
        path = RES / f"{synth}_{_slug(loss)}_{method}.pkl"
        if not path.exists():
            return []
        try:
            with path.open("rb") as fh:
                return pickle.load(fh).get("trials", [])
        except Exception:
            return []

    def load_scores(synth, loss, method):
        """Returned (final-candidate) P-loss scores for one synth × loss × method cell."""
        trials = load_trials(synth, loss, method)
        return [s for s in (_returned_p_loss(t, method) for t in trials) if np.isfinite(s)]

    def load_visited_scores(synth, loss, method):
        """Best-ever-visited P-loss scores (oracle) for the same cell."""
        trials = load_trials(synth, loss, method)
        out = []
        for t in trials:
            pl = np.asarray(t.get("history_p_loss", []), dtype=float)
            if len(pl) and np.any(np.isfinite(pl)):
                out.append(float(np.nanmin(pl)))
        return out

    return load_scores, load_visited_scores


@app.cell
def _(mannwhitneyu, np, pd):
    def _cliffs_delta(a, b):
        a, b = np.asarray(a), np.asarray(b)
        count = sum(np.sum(ai > b) - np.sum(ai < b) for ai in a)
        return count / (len(a) * len(b))

    def npsk_ranks(groups_dict, alpha=0.05, negligible_delta=0.147):
        valid = {k: np.asarray(v, float) for k, v in groups_dict.items()
                 if len(v) > 0 and np.any(np.isfinite(v))}
        valid = {k: v[np.isfinite(v)] for k, v in valid.items() if len(v[np.isfinite(v)]) > 0}
        if len(valid) < 2:
            return {k: 1 for k in valid}
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
            _, p = mannwhitneyu(lv, rv, alternative="two-sided")
            delta = _cliffs_delta(lv, rv)
            if p < alpha and abs(delta) >= negligible_delta:
                return sk_split(names[:best_i]) + sk_split(names[best_i:])
            return [list(names)]

        partitions = sk_split(sorted_names)
        return {n: r for r, grp in enumerate(partitions, 1) for n in grp}

    def compute_ranks_table(synths, losses, load_fn):
        rows = []
        for synth in synths:
            groups = {loss: load_fn(synth, loss) for loss in losses}
            groups = {k: v for k, v in groups.items() if v}
            if len(groups) < 2:
                row = {"synth": synth}
                for loss in losses:
                    row[loss] = "—" if loss not in groups else "solo"
                rows.append(row)
                continue
            ranks = npsk_ranks(groups)
            row = {"synth": synth}
            for loss in losses:
                row[loss] = int(ranks[loss]) if loss in ranks else "—"
            rows.append(row)
        return pd.DataFrame(rows).set_index("synth")

    return (compute_ranks_table,)


@app.cell
def _(METHODS, mo):
    method_pick = mo.ui.dropdown(options=METHODS, value=METHODS[0], label="Method")
    method_pick
    return (method_pick,)


@app.cell
def _(LOSSES, SYNTHS, load_scores, method_pick, mo, pd):
    # Data availability matrix for the selected method.
    _rows = []
    for _synth in SYNTHS:
        _row = {"synth": _synth}
        for _loss in LOSSES:
            _scores = load_scores(_synth, _loss, method_pick.value)
            _row[_loss] = len(_scores)
        _rows.append(_row)
    _avail_df = pd.DataFrame(_rows).set_index("synth")
    _avail_df.index.name = "Synth \\ Loss"

    def _mark(v):
        if v == 0:
            return f"✗ {v}"
        if v < 150:
            return f"⚠ {v}"
        return f"✓ {v}"

    _marked_df = _avail_df.map(_mark)
    _marked_df.index.name = "Synth \\ Loss"

    mo.vstack([
        mo.md(f"## Data availability — {method_pick.value} (n trials per cell)"),
        mo.md("✓ ≥ 150 trials  |  ⚠ < 150  |  ✗ = missing"),
        mo.ui.table(_marked_df.reset_index(), selection=None, label="Data availability"),
    ])
    return


@app.cell
def _(LOSSES, SYNTHS, compute_ranks_table, load_scores, method_pick, mo):
    computed_ranks = compute_ranks_table(
        SYNTHS, LOSSES, lambda s, l: load_scores(s, l, method_pick.value)
    )
    computed_ranks.index.name = "Synth \\ Loss"

    mo.vstack([
        mo.md(f"## Computed NPSK P-loss ranks — {method_pick.value}"),
        mo.md("Rank 1 = best (lowest expected P-Loss).  Ties = same rank number.  `—` = no data, `solo` = only one loss has data."),
        mo.ui.table(computed_ranks.reset_index(), selection=None, label="Computed ranks"),
    ])
    return (computed_ranks,)


@app.cell
def _(PUBLISHED_RANKS, computed_ranks, method_pick, mo, pd):
    # Rank-1 only: does the method correctly identify the *best* loss per synth?
    # (the headline question — exact ordering of ranks 2-4 matters far less)
    _rows = []
    for _synth, _pub in PUBLISHED_RANKS.items():
        _pub_best = sorted(l for l, r in _pub.items() if r == 1)
        if _synth in computed_ranks.index:
            _comp_best = sorted(
                l for l in _pub if computed_ranks.loc[_synth, l] == 1
            )
        else:
            _comp_best = []
        _comp_str = ", ".join(_comp_best) if _comp_best else "—"
        _match = (
            "—" if not _comp_best
            else ("✓" if set(_comp_best) & set(_pub_best) else "✗")
        )
        _rows.append({
            "synth": _synth,
            "published rank-1": ", ".join(_pub_best),
            f"{method_pick.value} rank-1": _comp_str,
            "match": _match,
        })

    _cmp_df = pd.DataFrame(_rows)
    _n_ok = (_cmp_df["match"] == "✓").sum()
    _n_miss = (_cmp_df["match"] == "—").sum()
    _n_fail = (_cmp_df["match"] == "✗").sum()
    _n_total = len(_cmp_df)

    _verdict = (
        f"**{_n_ok}/{_n_total} synths: {method_pick.value} picks the published best loss, "
        f"{_n_miss} missing, {_n_fail} disagreements.**"
        if _n_fail == 0
        else f"⚠ **{_n_fail}/{_n_total} disagreements** — {_n_ok} match, {_n_miss} missing."
    )

    mo.vstack([
        mo.md(f"## Rank-1 comparison — {method_pick.value}"),
        mo.md("_Which loss does each method correctly identify as **best** (rank 1) for each synth?_"),
        mo.md(_verdict),
        mo.ui.table(_cmp_df, selection=None, label="Rank-1 comparison"),
        mo.md(
            "_Note: published ranks come from GD-based P-Loss NPSK studies in the "
            "source papers — there is no published ranking for CMA-ES/RandomSearch. "
            "This shows whether **the same loss is identified as best** regardless of "
            "which search method is used; disagreements may reflect a method-specific "
            "search bias rather than a replication failure._"
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # Cross-method comparison

    For a chosen synth × loss cell, compare all search methods
    head-to-head — same idea as `findings.py`'s per-synth stats table, but
    with the loss function held fixed so the comparison is apples-to-apples.

    **Returned** = the method's final candidate (last GD step / best-found
    for black-box methods). **Visited** = best P-loss ever sampled (oracle
    upper bound on what the method *could* have returned). The gap between
    them is the "deception" a method pays for not ending where it started best.
    """)
    return


@app.cell
def _(LOSSES, SYNTHS, mo):
    cmp_synth_pick = mo.ui.dropdown(options=SYNTHS, value=SYNTHS[0], label="Synth")
    cmp_loss_pick = mo.ui.dropdown(options=LOSSES, value=LOSSES[0], label="Loss")
    mo.hstack([cmp_synth_pick, cmp_loss_pick])
    return cmp_loss_pick, cmp_synth_pick


@app.cell
def _(
    METHODS,
    cmp_loss_pick,
    cmp_synth_pick,
    load_scores,
    load_visited_scores,
    mo,
    np,
    pd,
):
    def _method_stats(synth, loss):
        rows = []
        for m in METHODS:
            ret = load_scores(synth, loss, m)
            vis = load_visited_scores(synth, loss, m)
            if not ret or not vis:
                continue
            r_med, v_med = float(np.median(ret)), float(np.median(vis))
            rows.append({
                "method": m,
                "n": len(ret),
                "returned (median)": round(r_med, 4),
                "visited (median)": round(v_med, 4),
                "deception gap": round(r_med - v_med, 4),
            })
        return pd.DataFrame(rows)

    _df = _method_stats(cmp_synth_pick.value, cmp_loss_pick.value)
    _out = (
        mo.ui.table(_df, selection=None, label="Method comparison")
        if not _df.empty else mo.md("_no data for this synth × loss cell yet_")
    )

    mo.vstack([
        mo.md(f"## {cmp_synth_pick.value} / {cmp_loss_pick.value} — methods compared"),
        _out,
    ])
    return


@app.cell
def _(LOSSES, METHODS, SYNTHS, load_scores, mo, np, pd):
    # Overview: for every synth, which method achieves the lowest median
    # returned P-loss, aggregated across all losses (best-loss-per-synth view).
    _rows = []
    for _synth in SYNTHS:
        _row = {"synth": _synth}
        for _m in METHODS:
            _all_scores = []
            for _loss in LOSSES:
                _all_scores.extend(load_scores(_synth, _loss, _m))
            _row[_m] = round(float(np.median(_all_scores)), 4) if _all_scores else float("nan")
        _rows.append(_row)

    _df = pd.DataFrame(_rows).set_index("synth")
    _df.index.name = "Synth \\ Method"

    def _fmt(v):
        return "—" if np.isnan(v) else f"{v:.4f}"

    def _bold_min(row):
        finite = {k: v for k, v in row.items() if not np.isnan(v)}
        if not finite:
            return {k: _fmt(v) for k, v in row.items()}
        best = min(finite.values())
        return {k: (f"**{_fmt(v)}**" if v == best else _fmt(v)) for k, v in row.items()}

    _fmt_df = _df.apply(_bold_min, axis=1, result_type="expand")
    _fmt_df.index.name = "Synth \\ Method"

    mo.vstack([
        mo.md("## Overview — median returned P-loss per method, pooled across all 4 losses"),
        mo.md("Bold = best method per synth (lower = better). Pooling across losses gives "
              "a coarse 'which method wins on this loss landscape' view; use the cell "
              "comparison above for a fixed-loss apples-to-apples read."),
        mo.ui.table(_fmt_df.reset_index(), selection=None, label="Pooled median returned P-loss"),
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
