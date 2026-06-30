import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import marimo as mo

    from paper_experiments.make_verification_report import METHODS, build_html
    return METHODS, build_html, mo


@app.cell
def _(mo):
    mo.md("""
    # Verification report explorer

    Pick a search method below — the rank-comparison report (NPSK P-loss
    rankings vs. the published IEEE/ISMIR in-domain rankings) is rendered
    inline, so there's no need to open separate `*_report.html` files.
    """)
    return


@app.cell
def _(METHODS, mo):
    method_pick = mo.ui.dropdown(options=METHODS, value=METHODS[0], label="Method")
    method_pick
    return (method_pick,)


@app.cell
def _(build_html, method_pick, mo):
    _html, _stats = build_html(method_pick.value)
    mo.iframe(_html)
    return


if __name__ == "__main__":
    app.run()
