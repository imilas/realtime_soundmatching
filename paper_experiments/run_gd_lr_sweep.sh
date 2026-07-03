#!/usr/bin/env bash
# Run GD experiments for a single learning-rate set across all synths × all GD losses.
#
# Usage:
#   bash paper_experiments/run_gd_lr_sweep.sh <lr>
#
# Example (run one set at a time):
#   bash paper_experiments/run_gd_lr_sweep.sh 0.0225
#   bash paper_experiments/run_gd_lr_sweep.sh 0.01125
#   bash paper_experiments/run_gd_lr_sweep.sh 0.09
#
# Results land in paper_experiments/results/{synth}_{loss}_GD_lr{lr}.pkl
# and are automatically visible in the trajectory/report notebooks.

set -euo pipefail

LR="${1:-}"
if [[ -z "$LR" ]]; then
    echo "Usage: $0 <learning_rate>"
    echo "  e.g. $0 0.0225"
    exit 1
fi

SYNTHS=(
    bandpass_noise
    am_noise
    add_sinesaw
    sine_mod_saw
    chirplet
    dx7_alg1
    dx7_alg2
    dx7_alg3
)

GD_LOSSES=(L1_Spec SIMSE_Spec DTW_Envelope JTFS)

JOBS=()
for synth in "${SYNTHS[@]}"; do
    for loss in "${GD_LOSSES[@]}"; do
        JOBS+=("$synth:::$loss")
    done
done

TOTAL=${#JOBS[@]}
echo "=== GD LR sweep: lr=$LR | $TOTAL jobs | $(date) ==="

export PATH="/cshome/asalimi/.conda/envs/soundmatch/bin:$PATH"
export PKG_CONFIG_PATH="/cshome/asalimi/.conda/envs/soundmatch/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export JAX_PLATFORM_NAME=cpu

# 2 cores per job × 10 parallel = 20 cores total
printf '%s\n' "${JOBS[@]}" | xargs -P 10 -I{} bash -c '
    job="$1"; lr="$2"
    synth="${job%%:::*}"
    loss="${job##*:::}"
    /cshome/asalimi/.conda/envs/soundmatch/bin/python paper_experiments/run_paper.py \
        --synth "$synth" \
        --method GD \
        --loss "$loss" \
        --lr "$lr" \
        --trials 200 \
        --budget 500 \
        >> "paper_experiments/results/gd_lr${lr}_sweep.log" 2>&1
' _ {} "$LR"

echo "=== Done: lr=$LR | $(date) ==="
