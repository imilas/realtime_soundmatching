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
    import matplotlib.patches as mpatches
    import numpy as np
    import pandas as pd
    from scipy.stats import mannwhitneyu

    RES = Path(__file__).parent / "results"

    LOSSES = ["SIMSE_Spec", "L1_Spec", "JTFS", "DTW_Envelope"]

    # In-domain synths: scenarios for which GD results should exist with all 4 losses.
    # OOD paper (loss_navigation_sound_matching/main.tex, Table 1, P-Loss columns),
    # giving us published ground-truth rankings to compare against.
    # "_v1" variants use the exact parameter ranges from the old in-domain paper
    # (ddsp_loss_comparison/) and are used to verify replication.
    # Matches make_verification_report.py's SYNTHS = list(PUBLISHED): only synths
    # with both replication data and a published P-Loss rank to compare against.
    SYNTHS = [
        "bandpass_noise",
        "add_sinesaw",
        "am_noise",
        "sine_mod_saw",
        "chirplet",
    ]

    # Published P-Loss NPSK rankings.
    # IEEE 2025 in-domain paper, Table III, P-Loss column (4 main synths).
    # OOD paper, Table 1, P-Loss column (in-domain chirplet scenarios).
    # Rank 1 = best (lowest expected P-Loss); ties mean NPSK could not distinguish.
    # Note: bandpass_noise maps to paper's BP-Noise (same DSP, correct ranges);
    #       add_sinesaw / am_noise / sine_mod_saw match the paper listings directly.
    PUBLISHED_RANKS = {
        # IEEE 2025 in-domain paper, Table III
        "bandpass_noise": {
            "SIMSE_Spec":   1,   # tied with L1 (both *1 in paper)
            "L1_Spec":      1,   # tied with SIMSE
            "JTFS":         4,
            "DTW_Envelope": 3,
        },
        "add_sinesaw": {
            "SIMSE_Spec":   4,
            "L1_Spec":      2,
            "JTFS":         1,
            "DTW_Envelope": 3,
        },
        "am_noise": {
            "SIMSE_Spec":   4,
            "L1_Spec":      2,
            "JTFS":         3,
            "DTW_Envelope": 1,
        },
        "sine_mod_saw": {
            "SIMSE_Spec":   2,
            "L1_Spec":      3,
            "JTFS":         4,
            "DTW_Envelope": 1,
        },
        # ISMIR paper, Table 1 (in-domain, chirplet only)
        "chirplet": {
            "SIMSE_Spec":   3,
            "L1_Spec":      1,
            "JTFS":         1,
            "DTW_Envelope": 4,
        },
    }

    # Expected best loss per synth (from IEEE 2025 in-domain paper and OOD paper).
    # Used for synths where we don't have full published rank tables.
    # _v1 variants replicate the old in-domain paper (ddsp_loss_comparison/) synths exactly;
    # their expected best matches the current-paper findings for the same synthesis programs.
    EXPECTED_BEST = {
        "bandpass_noise":    "SIMSE_Spec",
        "am_noise":          "DTW_Envelope",
        "add_sinesaw":       "SIMSE_Spec",
        "chirplet":          "JTFS",
        "sine_mod_saw":      "JTFS",
        "bandpass_noise": "SIMSE_Spec",
    }
    return (
        EXPECTED_BEST,
        LOSSES,
        PUBLISHED_RANKS,
        RES,
        SYNTHS,
        mannwhitneyu,
        mo,
        np,
        pd,
        pickle,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # GD verification: P-loss NPSK rankings vs published results

    This notebook verifies that the current gradient-descent (GD) implementation
    produces P-loss NPSK rankings that agree with previously published results.

    **Why this matters:** The new benchmark compares GD against black-box methods.
    That comparison is only valid if GD's behaviour has not changed relative to
    the prior in-domain work.  Here we re-run the same 4-loss comparison that the
    and check that the NPSK rankings match Table 1 of that paper.

    ---

    ## Reference: published P-loss ranks

    ### IEEE 2025 in-domain paper, Table III

    | Synth | SIMSE\_Spec | L1\_Spec | JTFS | DTW\_Envelope |
    |---|---|---|---|---|
    | `bandpass_noise` (BP-Noise) | **1** (tied) | **1** (tied) | 4 | 3 |
    | `add_sinesaw` (Add-SineSaw) | 4 | 2 | **1** | 3 |
    | `am_noise` (Noise-AM) | 4 | 2 | 3 | **1** |
    | `sine_mod_saw` (SineSaw-AM) | 2 | 3 | 4 | **1** |

    ### OOD paper, Table 1 (in-domain scenarios)

    | Synth | SIMSE\_Spec | L1\_Spec | JTFS | DTW\_Envelope |
    |---|---|---|---|---|
    | `chirplet` (Chirp: no delay) | 3 | 2 | **1** | 4 |

    Rank 1 = best.  A tie (same number) means NPSK could not distinguish the
    two distributions at α = 0.05, |Cliff's δ| ≥ 0.147.

    **Note:** `bandpass_noise` uses the paper's correct BP-Noise ranges [50,1000]/[1,120].
    The current `bandpass_noise.dsp` has different ranges and its PKL results are NOT
    comparable to the paper.  Run new GD experiments with `bandpass_noise` first.

    ---

    ## How to run missing L1\_Spec experiments

    L1\_Spec was added after the original GD runs.  If the PKL file for a synth/loss
    pair is missing the notebook will flag it.  To generate the results:

    ```bash
    conda activate soundmatch
    source experiment_scripts/env_capped.sh

        python paper_experiments/run_paper.py \
            --synth $synth --loss L1_Spec --method GD --trials 200
    done
    ```

    To run all 4 losses on the old-paper replication synths (`_v1`):

    ```bash
        for loss in SIMSE_Spec L1_Spec JTFS DTW_Envelope; do
            python paper_experiments/run_paper.py \
                --synth $synth --loss $loss --method GD --trials 200
        done
    done
    ```
    """)
    return


@app.cell
def _(LOSSES, RES, SYNTHS, mo, np, pd, pickle):
    def _returned_p_loss(trial, method="GD"):
        ploss = np.asarray(trial.get("history_p_loss", []), dtype=float)
        if len(ploss) == 0:
            return float("nan")
        if method == "GD":
            return float(ploss[-1])
        audio = np.asarray(trial.get("history_audio_loss", []), dtype=float)
        return float(ploss[int(np.nanargmin(audio))]) if len(audio) else float("nan")

    def load_gd_scores(synth: str, loss: str) -> list[float]:
        import re
        slug = re.sub(r"[^A-Za-z0-9]+", "_", loss).strip("_")
        path = RES / f"{synth}_{slug}_GD.pkl"
        if not path.exists():
            return []
        try:
            with path.open("rb") as fh:
                trials = pickle.load(fh).get("trials", [])
        except Exception:
            return []  # truncated / corrupt file
        return [s for s in (_returned_p_loss(t) for t in trials) if np.isfinite(s)]

    # Data availability matrix
    avail_rows = []
    for _synth in SYNTHS:
        row = {"synth": _synth}
        for _loss in LOSSES:
            scores = load_gd_scores(_synth, _loss)
            row[_loss] = len(scores) if scores else 0
        avail_rows.append(row)

    avail_df = pd.DataFrame(avail_rows).set_index("synth")
    avail_df.index.name = "Synth \\ Loss"

    def _mark(v):
        if v == 0:
            return f"✗ {v}"
        if v < 150:
            return f"⚠ {v}"
        return f"✓ {v}"

    _marked_df = avail_df.map(_mark)
    _marked_df.index.name = "Synth \\ Loss"
    mo.vstack([
        mo.md("## Data availability (n trials per cell)"),
        mo.md("✓ ≥ 150 trials  |  ⚠ < 150  |  ✗ = missing (run experiments above)"),
        mo.ui.table(_marked_df.reset_index(), selection=None, label="Data availability"),
    ])
    return (load_gd_scores,)


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
def _(LOSSES, SYNTHS, compute_ranks_table, load_gd_scores, mo):
    computed_ranks = compute_ranks_table(SYNTHS, LOSSES, load_gd_scores)
    computed_ranks.index.name = "Synth \\ Loss"

    mo.vstack([
        mo.md("## Computed NPSK P-loss ranks (current GD results)"),
        mo.md("Rank 1 = best (lowest expected P-Loss).  Ties = same rank number.  `—` = no data."),
        mo.ui.table(computed_ranks.reset_index(), selection=None, label="Computed ranks"),
    ])
    return (computed_ranks,)


@app.cell
def _(PUBLISHED_RANKS, computed_ranks, mo, pd):
    # Rank-1 only: does GD correctly identify the *best* loss per synth?
    # (the headline question — exact ordering of ranks 2-4 matters far less)
    _comparison_rows = []
    for _synth, _pub in PUBLISHED_RANKS.items():
        _pub_best = sorted(l for l, r in _pub.items() if r == 1)
        if _synth in computed_ranks.index:
            _comp_best = sorted(l for l in _pub if computed_ranks.loc[_synth, l] == 1)
        else:
            _comp_best = []
        _comp_str = ", ".join(_comp_best) if _comp_best else "—"
        _match = (
            "—" if not _comp_best
            else ("✓" if set(_comp_best) & set(_pub_best) else "✗")
        )
        _comparison_rows.append({
            "synth": _synth,
            "published rank-1": ", ".join(_pub_best),
            "GD rank-1": _comp_str,
            "match": _match,
        })

    _cmp_df = pd.DataFrame(_comparison_rows)
    _n_ok = (_cmp_df["match"] == "✓").sum()
    _n_miss = (_cmp_df["match"] == "—").sum()
    _n_fail = (_cmp_df["match"] == "✗").sum()

    _n_total = len(_cmp_df)
    _verdict = (
        f"**{_n_ok}/{_n_total} synths: GD picks the published best loss, "
        f"{_n_miss} missing, {_n_fail} disagreements.**"
        if _n_fail == 0
        else f"⚠ **{_n_fail}/{_n_total} disagreements** — investigate before claiming replication."
    )

    mo.vstack([
        mo.md("## Rank-1 comparison: computed vs published"),
        mo.md("_Which loss does GD correctly identify as **best** (rank 1) for each synth?_"),
        mo.md(_verdict),
        mo.ui.table(_cmp_df, selection=None, label="Rank-1 comparison"),
        mo.md(r"""
    **Notes:**

      rankings from the OOD/ISMIR paper (Table 1, columns "Chirp: no delay" and "Chirp: pulsating");
      the rest come from the IEEE 2025 in-domain paper (Table III).
    - The OOD paper ran **300 trials**; current results use **200 trials**.
      Minor rank differences at boundaries are expected; the top-rank winner should be identical.
    - `L1_Spec` was not in the original result set — run the experiments above if
      cells show `—`.
        """),
    ])
    return


@app.cell
def _(EXPECTED_BEST, LOSSES, SYNTHS, computed_ranks, mo, pd):
    # For synths without full published rank tables, show whether the expected
    # best loss achieves rank 1. Includes the _v1 replication synths.
    _check_rows = []
    for _synth in _no_pub:
        _expected = EXPECTED_BEST.get(_synth, "?")
        if _synth not in computed_ranks.index:
            _check_rows.append({
                "synth": _synth,
                "expected best": _expected,
                "computed rank of expected best": "—",
                "verdict": "no data",
            })
            continue
        _rank_of_best = computed_ranks.loc[_synth, _expected] if _expected in LOSSES else "—"
        _verdict_str = "✓ rank 1" if _rank_of_best == 1 else (
            "— missing" if _rank_of_best == "—" else f"✗ rank {_rank_of_best}"
        )
        _check_rows.append({
            "synth": _synth,
            "expected best": _expected,
            "computed rank of expected best": _rank_of_best,
            "verdict": _verdict_str,
        })

    mo.vstack([
        mo.md("## Expected-best check (synths without a full published table)"),
        mo.md("For these synths the prior papers give only the winning loss, not a full rank table."),
        mo.ui.table(pd.DataFrame(_check_rows), selection=None, label="Expected-best check"),
    ])
    return


@app.cell
def _(LOSSES, SYNTHS, load_gd_scores, mo, np, plt):
    # Bootstrap violin plot: bootstrapped median P-loss per loss function × synth.
    def _bootstrap_medians(scores, n_boot=1000, seed=0):
        rng = np.random.default_rng(seed)
        arr = np.asarray(scores, float)
        return np.array([np.median(rng.choice(arr, size=len(arr), replace=True))
                         for _ in range(n_boot)])

    _COLORS = {
        "SIMSE_Spec":    "#1f77b4",
        "L1_Spec":       "#ff7f0e",
        "JTFS":          "#2ca02c",
        "DTW_Envelope":  "#d62728",
    }

    _ncols = 4
    _nrows = (len(SYNTHS) + _ncols - 1) // _ncols
    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(14, 3.5 * _nrows), squeeze=False)

    for _idx, _synth in enumerate(SYNTHS):
        _ax = _axes[_idx // _ncols][_idx % _ncols]
        _groups = {l: load_gd_scores(_synth, l) for l in LOSSES}
        _groups = {l: s for l, s in _groups.items() if s}
        if not _groups:
            _ax.set_visible(False)
            continue
        _vdata, _pos, _cols, _labs = [], [], [], []
        for _pi, _l in enumerate(l for l in LOSSES if l in _groups):
            _vdata.append(_bootstrap_medians(_groups[_l]))
            _pos.append(_pi)
            _cols.append(_COLORS[_l])
            _labs.append(_l.replace("_", "\n"))
        _vp = _ax.violinplot(_vdata, positions=_pos, showmedians=True, widths=0.7)
        for _body, _c in zip(_vp["bodies"], _cols):
            _body.set_facecolor(_c)
            _body.set_alpha(0.75)
        for _part in ("cmedians", "cmins", "cmaxes", "cbars"):
            _vp[_part].set_color("black")
            _vp[_part].set_linewidth(0.8)
        _ax.set_xticks(_pos)
        _ax.set_xticklabels(_labs, fontsize=7)
        _ax.set_title(_synth, fontsize=8)
        _ax.set_ylabel("bootstrapped median\nreturned P-loss", fontsize=7)
        _ax.tick_params(axis="y", labelsize=7)

    for _idx in range(len(SYNTHS), _nrows * _ncols):
        _axes[_idx // _ncols][_idx % _ncols].set_visible(False)

    _fig.suptitle(
        "GD: bootstrapped median returned P-loss per loss function\n"
        "(lower = better; color = loss identity)",
        fontsize=9, y=1.01,
    )
    _fig.tight_layout()
    _out = mo.as_html(_fig)
    plt.close(_fig)

    mo.vstack([
        mo.md("## Bootstrapped P-loss distributions (GD only)"),
        _out,
    ])
    return


@app.cell
def _(LOSSES, SYNTHS, load_gd_scores, mo, np, pd):
    # Median table for quick reference.
    _rows = []
    for _synth in SYNTHS:
        _row = {"synth": _synth}
        for _loss in LOSSES:
            _scores = load_gd_scores(_synth, _loss)
            _row[_loss] = round(float(np.median(_scores)), 4) if _scores else float("nan")
        _rows.append(_row)
    _med_df = pd.DataFrame(_rows).set_index("synth")
    _med_df.index.name = "Synth \\ Loss"

    def _fmt(v):
        if np.isnan(v):
            return "—"
        return f"{v:.4f}"

    def _bold_min(row):
        finite = {k: v for k, v in row.items() if not np.isnan(v)}
        if not finite:
            return {k: _fmt(v) for k, v in row.items()}
        best = min(finite.values())
        return {k: (f"**{_fmt(v)}**" if v == best else _fmt(v)) for k, v in row.items()}

    _fmt_df = _med_df.apply(_bold_min, axis=1, result_type="expand")
    _fmt_df.index.name = "Synth \\ Loss"

    mo.vstack([
        mo.md("## Median returned P-loss (GD, all synths × losses)"),
        mo.md("Bold = best in row."),
        mo.ui.table(_fmt_df.reset_index(), selection=None, label="Median returned P-loss"),
    ])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
