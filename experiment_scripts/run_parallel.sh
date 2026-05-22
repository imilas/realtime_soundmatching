#!/bin/bash
# Example: run all synth/method combos in parallel using GNU parallel.
# Requires: sudo pacman -S parallel  (or apt install parallel)
#
# -j controls max concurrent jobs. A safe default is one per physical core.
# Each job is one (synth, method) cell — no shared state, no race conditions.
# QL trials within a cell still run sequentially (Q-table must carry over).

source .venv/bin/activate

SYNTHS="bandpass_noise am_noise add_sinesaw"
METHODS="HillClimber RandomSearch CMA-ES BO QL"
TRIALS=10
BUDGET=200
JOBS=6  # adjust to taste

parallel -j $JOBS \
    python paper_experiments/run_paper.py --synth {1} --method {2} --trials $TRIALS --budget $BUDGET \
    ::: $SYNTHS \
    ::: $METHODS
