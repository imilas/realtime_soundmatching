#!/bin/bash
# Full DX7 experiment run.
#   3 algorithms x 19 cells (4 losses x 4 methods + CLAP x 3 non-GD) = 57 cells
#   each: 300 trials x 500 eval budget.
# Resume-safe: run_paper.py loads existing trials and only runs the missing ones,
# so re-running this script continues an interrupted run.
#
# Usage:
#   bash experiment_scripts/run_dx7_fullrun.sh                # default 26 parallel jobs
#   JOBSCOUNT=20 bash experiment_scripts/run_dx7_fullrun.sh   # fewer jobs
set -u
cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.."
source experiment_scripts/env_capped.sh

TRIALS="${TRIALS:-300}"
BUDGET="${BUDGET:-500}"
JOBSCOUNT="${JOBSCOUNT:-20}"          # MY core footprint; keep <= 30, THREADS=1
CHUNK="${CHUNK:-25}"                  # trials per process before restart (frees JAX/XLA
                                      # memory-mappings; run_paper.py is resume-safe so each
                                      # batch tops the cell up to the next multiple of CHUNK).
                                      # Without this, a single 300-trial process accumulates
                                      # ~hundreds of per-trial XLA compilations and OOMs
                                      # ("Cannot allocate memory") despite ~1TB RAM free.

JOBS="$(mktemp)"
{
  for s in dx7_alg1 dx7_alg2 dx7_alg3; do
    for m in RandomSearch CMA-ES LES; do
      for l in SIMSE_Spec L1_Spec JTFS DTW_Envelope CLAP; do echo "$s $l $m"; done
    done
    for l in SIMSE_Spec L1_Spec JTFS DTW_Envelope; do echo "$s $l GD"; done   # GD: no CLAP
  done
} >| "$JOBS"

echo "[dx7 fullrun] start $(date) — $(wc -l < "$JOBS") cells, trials=$TRIALS budget=$BUDGET jobs=$JOBSCOUNT chunk=$CHUNK"
xargs -P "$JOBSCOUNT" -a "$JOBS" -L 1 bash -c '
  s="$1"; l="$2"; m="$3"
  for tgt in $(seq '"$CHUNK"' '"$CHUNK"' '"$TRIALS"') '"$TRIALS"'; do
    "$PY" paper_experiments/run_paper.py --synth "$s" --loss "$l" --method "$m" \
      --trials "$tgt" --budget '"$BUDGET"' || true
  done
' _
echo "[dx7 fullrun] done  $(date)"
