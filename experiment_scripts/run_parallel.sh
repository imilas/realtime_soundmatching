#!/bin/bash
# Run all synth/method combinations in parallel.
# Uses GNU parallel if available, otherwise bash background jobs.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# Activate environment
if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" != "base" ]; then
    : # already active
elif [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

SYNTHS="bandpass_noise am_noise add_sinesaw"
METHODS="HillClimber RandomSearch CMA-ES BO QL"
TRIALS=200
BUDGET=200
JOBS=15

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --trials N       Number of trials per synth/method cell (default: $TRIALS)
  --budget N       Evaluation budget per trial (default: $BUDGET)
  --jobs N         Maximum parallel jobs (default: $JOBS)
  --synths "..."   Space-separated synth list (default: "$SYNTHS")
  --methods "..."  Space-separated method list (default: "$METHODS")
  -h, --help       Show this help
EOF
}

while (($#)); do
    case "$1" in
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --budget)
            BUDGET="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --synths)
            SYNTHS="$2"
            shift 2
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

echo "Synths:  $SYNTHS"
echo "Methods: $METHODS"
echo "Trials:  $TRIALS"
echo "Budget:  $BUDGET"
echo "Jobs:    $JOBS"

mkdir -p paper_experiments/results

if command -v parallel &>/dev/null; then
    echo "Running with GNU parallel (-j $JOBS)..."
    parallel -j "$JOBS" \
        python paper_experiments/run_paper.py --synth {1} --method {2} --trials "$TRIALS" --budget "$BUDGET" \
        ::: $SYNTHS \
        ::: $METHODS
else
    echo "GNU parallel not found — using bash background jobs (max $JOBS at a time)..."
    running=0
    for synth in $SYNTHS; do
        for method in $METHODS; do
            python paper_experiments/run_paper.py \
                --synth "$synth" --method "$method" \
                --trials "$TRIALS" --budget "$BUDGET" &
            running=$((running + 1))
            if [ "$running" -ge "$JOBS" ]; then
                wait -n 2>/dev/null || wait
                running=$((running - 1))
            fi
        done
    done
    wait
fi

echo "All done. Results in paper_experiments/results/"
