import marimo as mo


def make_slider(params):
    return {
        name: mo.ui.slider(
            start=p.min_val,
            stop=p.max_val,
            value=p.default,
            step=p.step,
            label=name,
            show_value=True,
        )
        for name, p in params.items()
    }
