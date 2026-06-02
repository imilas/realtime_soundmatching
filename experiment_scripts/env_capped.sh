#!/bin/bash
# Source this to get: soundmatch conda env + faust/sndfile toolchain on PATH,
# and a HARD cap on CPU threads (shared 128-core server — keep us <= 20 cores).
#
# Usage:
#   source experiment_scripts/env_capped.sh         # default 1 thread/proc
#   THREADS=4 source experiment_scripts/env_capped.sh
#   python paper_experiments/...py
#
# For multi-process runs keep (parallel jobs) * THREADS <= 20.

ENV="${SOUNDMATCH_ENV:-/cshome/asalimi/.conda/envs/soundmatch}"
THREADS="${THREADS:-1}"

export PATH="$HOME/.local/bin:$ENV/bin:$PATH"
export PKG_CONFIG_PATH="$ENV/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"

# BLAS / OpenMP — one knob each so numpy/scipy/sklearn don't fan out.
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"

# JAX/XLA on CPU otherwise grabs every core.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$THREADS"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl}"
mkdir -p "$MPLCONFIGDIR" 2>/dev/null

# numba cache dir (harmless; the learned pipeline uses scipy, not librosa/numba)
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/numba}"
mkdir -p "$NUMBA_CACHE_DIR" 2>/dev/null

export PY="$ENV/bin/python"
echo "[env_capped] THREADS=$THREADS  PY=$PY  (keep jobs*THREADS <= 20)"
