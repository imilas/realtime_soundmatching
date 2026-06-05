#!/bin/bash
# Run 300 GD trials per synth with a 200-evaluation budget.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

JOBS="${JOBS:-1}"
THREADS="${THREADS:-1}"
SYNTHS="${SYNTHS:-bandpass_noise am_noise add_sinesaw}"

exec bash experiment_scripts/run_parallel.sh \
    --trials 300 \
    --budget 200 \
    --jobs "$JOBS" \
    --threads "$THREADS" \
    --synths "$SYNTHS" \
    --methods "GD"
