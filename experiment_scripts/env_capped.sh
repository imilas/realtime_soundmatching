#!/bin/bash
# Source this to get: soundmatch conda env + faust/sndfile toolchain on PATH,
# and a HARD cap on CPU threads (shared 128-core server — keep us <= 30 cores).
#
# Usage:
#   source experiment_scripts/env_capped.sh         # default 1 thread/proc
#   THREADS=4 source experiment_scripts/env_capped.sh
#   python paper_experiments/...py
#
# For multi-process runs keep (parallel jobs) * THREADS <= 30.

ENV="${SOUNDMATCH_ENV:-/cshome/asalimi/.conda/envs/soundmatch}"
THREADS="${THREADS:-1}"

export PATH="$HOME/.local/bin:$ENV/bin:$PATH"
export PKG_CONFIG_PATH="$ENV/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"

# BLAS / OpenMP — one knob each so numpy/scipy/sklearn don't fan out.
export GOMP_SPINCOUNT=0      # prevent idle OpenMP threads from busy-waiting
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"

# JAX/XLA: disable multi-threaded Eigen (=true enables it — common footgun).
# TF inter/intra-op pools catch any remaining TF-backed threading.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export TF_NUM_INTEROP_THREADS="$THREADS"
export TF_NUM_INTRAOP_THREADS="$THREADS"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl}"
mkdir -p "$MPLCONFIGDIR" 2>/dev/null

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/numba}"
mkdir -p "$NUMBA_CACHE_DIR" 2>/dev/null

export PY="$ENV/bin/python"
echo "[env_capped] THREADS=$THREADS  PY=$PY  (keep jobs*THREADS <= 30)"
