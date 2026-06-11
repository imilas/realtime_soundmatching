#!/bin/bash
# Run the GD verification experiments: replicate the IEEE in-domain paper's
# 4 synths × 4 losses using the current project's GD code.
# Results feed into paper_experiments/gd_verification.py.
#
# Synth mapping (IEEE paper → current project):
#   BP-Noise     → bandpass_noise_v1  (lp_cut [50,1000], hp_cut [1,120])
#   Add-SineSaw  → add_sinesaw
#   Noise-AM     → am_noise
#   SineSaw-AM   → sine_mod_saw
#
# Usage:
#   bash experiment_scripts/run_verification.sh
#   bash experiment_scripts/run_verification.sh --jobs 5 --trials 200

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

JOBS=7
TRIALS=200
BUDGET=200

while (($#)); do
  case "$1" in
    --jobs)   JOBS="$2";   shift 2 ;;
    --trials) TRIALS="$2"; shift 2 ;;
    --budget) BUDGET="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--jobs N] [--trials N] [--budget N]"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

source experiment_scripts/env_capped.sh

THREAD_ENV=(
  OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  VECLIB_MAXIMUM_THREADS=1
  GOMP_SPINCOUNT=0
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
  TF_NUM_INTEROP_THREADS=1
  TF_NUM_INTRAOP_THREADS=1
  MPLCONFIGDIR="$MPLCONFIGDIR"
  NUMBA_CACHE_DIR="$NUMBA_CACHE_DIR"
)

SYNTHS="bandpass_noise_v1 add_sinesaw am_noise sine_mod_saw"
LOSSES="SIMSE_Spec L1_Spec JTFS DTW_Envelope"

JOBFILE=$(mktemp)
trap 'rm -f "$JOBFILE"' EXIT

for synth in $SYNTHS; do
  for loss in $LOSSES; do
    echo "$synth $loss"
  done
done > "$JOBFILE"

TOTAL=$(wc -l < "$JOBFILE")
echo "Verification cells: $TOTAL  (jobs=$JOBS, trials=$TRIALS, budget=$BUDGET)"
echo "Synths: $SYNTHS"
echo "Losses: $LOSSES"
echo ""

run_one() {
  synth="$1"; loss="$2"
  logfile="paper_experiments/results/${synth}_${loss}_GD.log"
  env "${THREAD_ENV[@]}" "$PY" paper_experiments/run_paper.py \
    --synth "$synth" --loss "$loss" --method GD \
    --trials "$TRIALS" --budget "$BUDGET" \
    > "$logfile" 2>&1
  status=$?
  last=$(tail -1 "$logfile" 2>/dev/null || true)
  echo "[$(date '+%H:%M:%S')] $status | $synth / $loss | $last"
}
export -f run_one
export PY TRIALS BUDGET
export "${THREAD_ENV[@]}"

xargs -a "$JOBFILE" -P "$JOBS" -L 1 bash -c 'run_one $1 $2' _

echo ""
echo "Done. Open the notebook to check rankings:"
echo "  marimo run paper_experiments/gd_verification.py"
