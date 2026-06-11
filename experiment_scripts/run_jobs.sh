#!/bin/bash
# Run paper experiments in parallel, hard-capped to JOBS concurrent processes.
#
# Each process is nominally single-threaded via BLAS/OpenMP/JAX/XLA env vars,
# but JAX's XLA JIT compiler and internal thread pools still use ~3-4 cores
# per process in practice.  Default JOBS=7 keeps total load ≤ 28 cores.
# Raise --jobs only if you have verified lighter load (e.g. non-GD methods).
#
# Usage:
#   bash experiment_scripts/run_jobs.sh                        # all cells
#   bash experiment_scripts/run_jobs.sh --jobs 10              # more parallel
#   bash experiment_scripts/run_jobs.sh --synths "chirplet am_noise"
#
# Each cell is: synth × loss × method.  Cells that already have >= TRIALS
# results are skipped automatically by run_paper.py.

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
JOBS=7
TRIALS=200
BUDGET=200
SYNTHS="bandpass_noise am_noise add_sinesaw sine_saw sine_mod_saw sine_mod_sine chirplet chirplet_pulse bandpass_noise_v1 am_noise_v1"
LOSSES="SIMSE_Spec DTW_Envelope JTFS L1_Spec"
METHODS="GD CMA-ES RandomSearch"

usage() {
  cat <<EOF
Usage: $0 [options]
  --jobs N        Max parallel processes (default: $JOBS). Each uses 1 thread.
  --trials N      Trials per cell (default: $TRIALS)
  --budget N      Eval budget per trial (default: $BUDGET)
  --synths "..."  Space-separated synth list
  --losses "..."  Space-separated loss list
  --methods "..." Space-separated method list
  -h, --help
EOF
}

while (($#)); do
  case "$1" in
    --jobs)    JOBS="$2";    shift 2 ;;
    --trials)  TRIALS="$2";  shift 2 ;;
    --budget)  BUDGET="$2";  shift 2 ;;
    --synths)  SYNTHS="$2";  shift 2 ;;
    --losses)  LOSSES="$2";  shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Environment — source once so PY and PATH are set, then re-export everything
# explicitly into each child via env(1) so inheritance is guaranteed.
# ---------------------------------------------------------------------------
source experiment_scripts/env_capped.sh

THREAD_ENV=(
  OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  VECLIB_MAXIMUM_THREADS=1
  GOMP_SPINCOUNT=0          # prevent idle OpenMP threads from busy-waiting
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
  TF_NUM_INTEROP_THREADS=1
  TF_NUM_INTRAOP_THREADS=1
  MPLCONFIGDIR="$MPLCONFIGDIR"
  NUMBA_CACHE_DIR="$NUMBA_CACHE_DIR"
)

# ---------------------------------------------------------------------------
# Build job list
# ---------------------------------------------------------------------------
JOBFILE=$(mktemp)
trap 'rm -f "$JOBFILE"' EXIT

for synth in $SYNTHS; do
  for loss in $LOSSES; do
    for method in $METHODS; do
      # GD only supports specific losses — skip silently if unsupported.
      if [[ "$method" == "GD" ]]; then
        case "$loss" in SIMSE_Spec|DTW_Envelope|JTFS|L1_Spec) ;; *) continue ;; esac
      fi
      echo "$synth $loss $method"
    done
  done
done > "$JOBFILE"

TOTAL=$(wc -l < "$JOBFILE")
echo "Jobs to run: $TOTAL  (jobs=$JOBS, trials=$TRIALS, budget=$BUDGET)"
echo ""

# ---------------------------------------------------------------------------
# Run with xargs -P (available on all Linux/macOS without GNU parallel)
# ---------------------------------------------------------------------------
run_one() {
  synth="$1"; loss="$2"; method="$3"
  logfile="paper_experiments/results/${synth}_${loss}_${method}.log"
  # run_paper.py is resume-safe (skips already-completed trials), so on a
  # crash (e.g. the occasional XLA/LLVM JIT "Unable to allocate section
  # memory" race under high parallelism) just retry — a fresh process
  # almost always succeeds.
  for attempt in 1 2 3; do
    env "${THREAD_ENV[@]}" "$PY" paper_experiments/run_paper.py \
      --synth "$synth" --loss "$loss" --method "$method" \
      --trials "$TRIALS" --budget "$BUDGET" \
      >> "$logfile" 2>&1
    status=$?
    if [[ $status -eq 0 ]]; then
      break
    fi
    echo "[$(date '+%H:%M:%S')] attempt $attempt failed (status=$status), retrying..." >> "$logfile"
  done
  last=$(tail -1 "$logfile" 2>/dev/null || true)
  echo "[$(date '+%H:%M:%S')] $status | $synth / $loss / $method | $last"
}
export -f run_one
export PY TRIALS BUDGET
export "${THREAD_ENV[@]}"

xargs -a "$JOBFILE" -P "$JOBS" -L 1 bash -c 'run_one $1 $2 $3' _

echo ""
echo "All done. Results in paper_experiments/results/"
