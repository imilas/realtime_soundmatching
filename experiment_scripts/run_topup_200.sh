#!/bin/bash
# Top up all experiment results to 200 trials / 500 eval budget.
# Fixes chirplet/SIMSE_Spec/GD (bad eval_budget=20) then resumes all combos.
# Resume-safe: the runner skips any (synth,loss,method) that already has >= 200 trials.

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

THREADS=1
JOBS=40
TRIALS=200
BUDGET=500

source experiment_scripts/env_capped.sh

SYNTHS="bandpass_noise am_noise add_sinesaw sine_mod_saw chirplet dx7_alg1 dx7_alg2 dx7_alg3 dx7_5op_am"
LOSSES="SIMSE_Spec L1_Spec JTFS DTW_Envelope CLAP"
BASE_METHODS="GD CMA-ES LES RandomSearch"
LR_VARIANTS="0.005625 0.01125 0.0225"

# Fix: chirplet/SIMSE_Spec/GD was run with eval_budget=20 — delete and re-run.
BAD_PKL="paper_experiments/results/chirplet_SIMSE_Spec_GD.pkl"
if [ -f "$BAD_PKL" ]; then
    echo "[fix] Deleting $BAD_PKL (eval_budget=20, should be $BUDGET)"
    rm "$BAD_PKL"
fi

commands=()
for synth in $SYNTHS; do
    for loss in $LOSSES; do
        for method in $BASE_METHODS; do
            # GD doesn't support CLAP
            if [ "$method" = "GD" ] && [ "$loss" = "CLAP" ]; then continue; fi
            printf -v cmd \
                'OMP_NUM_THREADS=%q OPENBLAS_NUM_THREADS=%q MKL_NUM_THREADS=%q NUMEXPR_NUM_THREADS=%q VECLIB_MAXIMUM_THREADS=%q NUMBA_NUM_THREADS=%q %q paper_experiments/run_paper.py --synth %q --loss %q --method %q --trials %q --budget %q' \
                "$THREADS" "$THREADS" "$THREADS" "$THREADS" "$THREADS" "$THREADS" \
                "$PY" "$synth" "$loss" "$method" "$TRIALS" "$BUDGET"
            commands+=("$cmd")
        done
        # GD learning-rate variants (no CLAP)
        if [ "$loss" = "CLAP" ]; then continue; fi
        for lr in $LR_VARIANTS; do
            printf -v cmd \
                'OMP_NUM_THREADS=%q OPENBLAS_NUM_THREADS=%q MKL_NUM_THREADS=%q NUMEXPR_NUM_THREADS=%q VECLIB_MAXIMUM_THREADS=%q NUMBA_NUM_THREADS=%q %q paper_experiments/run_paper.py --synth %q --loss %q --method GD --lr %q --trials %q --budget %q' \
                "$THREADS" "$THREADS" "$THREADS" "$THREADS" "$THREADS" "$THREADS" \
                "$PY" "$synth" "$loss" "$lr" "$TRIALS" "$BUDGET"
            commands+=("$cmd")
        done
    done
done

echo "Total jobs: ${#commands[@]}  (parallel: $JOBS)"
echo "Each job skips if already at $TRIALS trials."
echo ""

parallel -j "$JOBS" -- "${commands[@]}"

echo ""
echo "Done. Run python /tmp/check_experiments.py to verify."
