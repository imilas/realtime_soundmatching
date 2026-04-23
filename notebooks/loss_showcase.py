import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from utils.loss_functions import ALL_LOSSES

    return ALL_LOSSES, mo, np, plt


@app.cell
def _(mo):
    mo.md("""
    # Loss Function Showcase

    We generate two sine waves — a fixed **target** and a **candidate** whose frequency
    is swept across a range — and plot each loss function vs frequency offset.

    A good loss landscape for optimisation is smooth and has a clear minimum at the
    target frequency (dashed red line).
    """)
    return


@app.cell
def _(np):
    SAMPLE_RATE = 44100
    DURATION = 0.5  # seconds
    TARGET_FREQ = 440.0  # Hz
    SWEEP_FREQS = np.linspace(100, 1000, 200)

    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    target = np.sin(2 * np.pi * TARGET_FREQ * t)
    candidates = [np.sin(2 * np.pi * f * t) for f in SWEEP_FREQS]
    return SAMPLE_RATE, SWEEP_FREQS, TARGET_FREQ, candidates, target


@app.cell
def _(ALL_LOSSES, SAMPLE_RATE, candidates, target):
    loss_results = {}
    def _():
        for name, loss_fn in ALL_LOSSES.items():
            loss_results[name] = [loss_fn(target, c, SAMPLE_RATE) for c in candidates]
        return


    _()
    return (loss_results,)


@app.cell
def _(ALL_LOSSES, SWEEP_FREQS, TARGET_FREQ, loss_results, plt):
    n = len(ALL_LOSSES)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    for i, (name, values) in enumerate(loss_results.items()):
        ax = axes[i]
        ax.plot(SWEEP_FREQS, values, linewidth=1.5)
        ax.axvline(TARGET_FREQ, color="red", linestyle="dashed", alpha=0.6, label="target")
        ax.set_title(name, fontsize=14)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)

    # hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
