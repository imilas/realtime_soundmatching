#!/bin/bash
# Run a grid of paper experiments.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

SYNTHS="bandpass_noise am_noise add_sinesaw"
METHODS="HillClimber RandomSearch CMA-ES BO QL"
TRIALS=200
BUDGET=200
JOBS=2
THREADS=1
TIMING_FILE=""

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --trials N       Trials per synth/method cell (default: $TRIALS)
  --budget N       Evaluation budget per trial (default: $BUDGET)
  --jobs N         Parallel jobs (default: $JOBS)
  --threads N      Native CPU threads per job (default: $THREADS)
  --synths "..."   Space-separated synth list (default: "$SYNTHS")
  --methods "..."  Space-separated method list (default: "$METHODS")
  --timing-file P  Timing summary TSV path (default: timestamped file in results/)
  -h, --help       Show this help

Example:
  $0 --trials 10 --budget 50 --jobs 2
EOF
}

while (($#)); do
    case "$1" in
        --trials) TRIALS="$2"; shift 2 ;;
        --budget) BUDGET="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --synths) SYNTHS="$2"; shift 2 ;;
        --methods) METHODS="$2"; shift 2 ;;
        --timing-file) TIMING_FILE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

mkdir -p paper_experiments/results
if [ -z "$TIMING_FILE" ]; then
    TIMING_FILE="paper_experiments/results/run_timings_$(date +%Y%m%d_%H%M%S).tsv"
fi
: > "$TIMING_FILE"

echo "Synths:  $SYNTHS"
echo "Methods: $METHODS"
echo "Trials:  $TRIALS"
echo "Budget:  $BUDGET"
echo "Jobs:    $JOBS"
echo "Threads: $THREADS per job"
echo "Timing:  $TIMING_FILE"
echo ""

if ! command -v parallel >/dev/null 2>&1; then
    echo "moreutils parallel is required but was not found on PATH." >&2
    exit 1
fi

commands=()
for synth in $SYNTHS; do
    for method in $METHODS; do
        printf -v cmd \
            'start=$(date +%%s); echo "==> %q / %q"; OMP_NUM_THREADS=%q OPENBLAS_NUM_THREADS=%q MKL_NUM_THREADS=%q NUMEXPR_NUM_THREADS=%q VECLIB_MAXIMUM_THREADS=%q python paper_experiments/run_paper.py --synth %q --method %q --trials %q --budget %q; status=$?; end=$(date +%%s); elapsed=$((end - start)); printf "%%s\t%%s\t%%s\t%%s\n" %q %q "$elapsed" "$status" >> %q; exit "$status"' \
            "$synth" "$method" "$THREADS" "$THREADS" "$THREADS" "$THREADS" "$THREADS" "$synth" "$method" "$TRIALS" "$BUDGET" "$synth" "$method" "$TIMING_FILE"
        commands+=("$cmd")
    done
done

parallel_status=0
parallel -j "$JOBS" -- "${commands[@]}" || parallel_status=$?

echo ""
echo "Timing summary by method:"
printf "%-14s %5s %10s %12s %8s\n" "method" "cells" "total_s" "max_cell_s" "failed"
for method in $METHODS; do
    awk -F '\t' -v method="$method" '
        $2 == method {
            n += 1
            total += $3
            if ($3 > max) max = $3
            if ($4 != 0) failed += 1
        }
        END {
            if (n > 0) {
                printf "%-14s %5d %10d %12d %8d\n", method, n, total, max, failed
            }
        }
    ' "$TIMING_FILE"
done

echo "All done. Results in paper_experiments/results/"
echo "Timing details: $TIMING_FILE"
exit "$parallel_status"
