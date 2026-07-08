"""
Data loading for fr_v2.py. This file should rarely change —
keeping it separate from final_report_helper.py means editing
plot functions never triggers a data reload in marimo.
"""

from __future__ import annotations

import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

RES = Path(__file__).parent / "results"

_PLOT_FIELDS = {"method", "loss_name", "eval_budget", "best_p_loss",
                "history_p_loss", "history_audio_loss"}


def _load_one(synth: str, loss: str, method: str) -> list:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", loss).strip("_")
    pkl = RES / f"{synth}_{slug}_{method}.pkl"
    if not pkl.exists():
        return []
    try:
        data = pickle.load(open(pkl, "rb"))
        trials = data.get("trials", []) if isinstance(data, dict) else data
        return [{k: v for k, v in t.items() if k in _PLOT_FIELDS} for t in trials]
    except Exception:
        return []


def load_all_trials(
    synths: list[str],
    losses: list[str],
    methods: list[str],
    max_workers: int = 40,
) -> dict:
    """Load all (synth, loss, method) pkl files in parallel."""
    keys = [(s, l, m) for s in synths for l in losses for m in methods]
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_load_one, *zip(*keys)))
    return dict(zip(keys, results))
