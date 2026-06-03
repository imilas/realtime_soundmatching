#!/usr/bin/env bash
# Full paper experiment grid: synth x loss x method.
#
# Defaults: 100 trials, budget 400 evals per trial, 4 parallel cells, 1 thread per cell.
# Resumable: each cell writes its own pkl in paper_experiments/results/ and skips
# already-completed trials, so killing and re-running this script picks up where it left off.
#
# Override any default by exporting before the call, e.g.:
#   TRIALS=50 BUDGET=400 JOBS=8 ./paper_experiments/run_full_paper.sh
#
# To run only part of the grid, pass --synths / --losses / --methods through:
#   ./paper_experiments/run_full_paper.sh --losses JTFS SIMSE_Spec --methods GD CMA-ES

set -euo pipefail

cd "$(dirname "$0")/.."

# Activate the project venv (Python 3.10 + DawDreamer + JAX + kymatio).
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "ERROR: .venv not found. Create it per CLAUDE.md (Python 3.10) before running." >&2
  exit 1
fi

TRIALS="${TRIALS:-100}"
BUDGET="${BUDGET:-400}"
JOBS="${JOBS:-4}"
THREADS="${THREADS:-1}"

mkdir -p paper_experiments/results paper_experiments/logs
LOG="paper_experiments/logs/full_paper_$(date +%Y%m%d_%H%M%S).log"

echo "Writing combined log to $LOG"
echo "TRIALS=$TRIALS BUDGET=$BUDGET JOBS=$JOBS THREADS=$THREADS"

python experiment_scripts/run_loss_grid.py \
  --trials "$TRIALS" \
  --budget "$BUDGET" \
  --jobs "$JOBS" \
  --threads "$THREADS" \
  --synths all \
  --losses all \
  --methods all \
  "$@" 2>&1 | tee -a "$LOG"
