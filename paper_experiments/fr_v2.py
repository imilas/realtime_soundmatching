import marimo

__generated_with = "0.8.22"
app = marimo.App(width="full")


@app.cell
def _(__file__):
    import sys
    import importlib
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    import marimo as mo

    from paper_experiments.make_verification_report import (
        LOSSES_ALL,
        SYNTHS,
        SYNTH_LABELS,
        discover_methods,
    )
    return (
        LOSSES_ALL,
        Path,
        SYNTHS,
        SYNTH_LABELS,
        discover_methods,
        importlib,
        mo,
        sys,
    )


@app.cell
def _(SYNTHS, SYNTH_LABELS, discover_methods, mo):
    all_methods = discover_methods()
    synth_select = mo.ui.multiselect(
        options={SYNTH_LABELS[s]: s for s in SYNTHS},
        value=[SYNTH_LABELS[s] for s in SYNTHS],
        label="Synths",
    )
    method_select = mo.ui.multiselect(
        options=all_methods,
        value=all_methods,
        label="Methods",
    )
    mo.vstack([mo.md("### Controls"), synth_select, method_select])
    return all_methods, method_select, synth_select


@app.cell
def _(LOSSES_ALL, SYNTHS, all_methods):
    from paper_experiments.final_report_loader import load_all_trials
    trial_cache = load_all_trials(SYNTHS, list(LOSSES_ALL), all_methods)
    return load_all_trials, trial_cache


@app.cell
def _(
    LOSSES_ALL,
    SYNTHS,
    SYNTH_LABELS,
    all_methods,
    importlib,
    method_select,
    mo,
    synth_select,
    trial_cache,
):
    import paper_experiments.final_report_helper as frh
    importlib.reload(frh)
    selected_synths = [s for s in SYNTHS if s in synth_select.value] or list(SYNTHS)
    selected_methods = method_select.value or list(all_methods)
    colors = frh.build_method_colors(all_methods)

    fig = frh.sample_efficiency_plot(
        trial_cache,
        synths=selected_synths,
        losses=list(LOSSES_ALL),
        methods=selected_methods,
        synth_labels=SYNTH_LABELS,
        method_colors=colors,
    )
    mo.Html(frh.fig_to_html(fig, dpi=90, min_width=2400))
    return colors, fig, frh, selected_methods, selected_synths


@app.cell
def __(
    LOSSES_ALL,
    SYNTH_LABELS,
    colors,
    frh,
    importlib,
    mo,
    selected_methods,
    selected_synths,
    trial_cache,
):

    importlib.reload(frh)
    # selected_synths = [s for s in SYNTHS if s in synth_select.value] or list(SYNTHS)
    # selected_methods = method_select.value or list(all_methods)
    # colors = frh.build_method_colors(all_methods)

    fig2 = frh.returned_ploss_plot(
        trial_cache,
        synths=selected_synths,
        losses=list(LOSSES_ALL),
        methods=selected_methods,
        synth_labels=SYNTH_LABELS,
        method_colors=colors,
    )
    mo.Html(frh.fig_to_html(fig2, dpi=44, min_width=2000))

    return (fig2,)


@app.cell
def __(
    LOSSES_ALL,
    SYNTH_LABELS,
    colors,
    frh,
    importlib,
    mo,
    selected_methods,
    selected_synths,
    trial_cache,
):
    importlib.reload(frh)
    # selected_synths = [s for s in SYNTHS if s in synth_select.value] or list(SYNTHS)
    # selected_methods = method_select.value or list(all_methods)
    # colors = frh.build_method_colors(all_methods)

    fig3 = frh.audio_loss_plot(
        trial_cache,
        synths=selected_synths,
        losses=list(LOSSES_ALL),
        methods=selected_methods,
        synth_labels=SYNTH_LABELS,
        method_colors=colors,
    )
    mo.Html(frh.fig_to_html(fig3, dpi=80, min_width=2000))
    return (fig3,)


app._unparsable_cell(
    r"""
        importlib.reload(frh)
        selected_synths = [s for s in SYNTHS if s in synth_select.value] or list(SYNTHS)
        selected_methods = method_select.value or list(all_methods)
        colors = frh.build_method_colors(all_methods)

        fig4 = frh.audio_loss_best_so_far_plot(
            trial_cache,
            synths=selected_synths,
            losses=list(LOSSES_ALL),
            methods=selected_methods,
            synth_labels=SYNTH_LABELS,
            method_colors=colors,
        )
        mo.Html(frh.fig_to_html(fig4, dpi=110, min_width=2000))
    """,
    name="__"
)


if __name__ == "__main__":
    app.run()
