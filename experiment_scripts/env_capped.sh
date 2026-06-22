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
# Extra packages that can't be installed into the read-only env site-packages
# (e.g. jinja2, needed by pandas .style for the gd_verification notebook).
export PYTHONPATH="$HOME/.local/soundmatch-extra:${PYTHONPATH:-}"
export PKG_CONFIG_PATH="$ENV/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"

# BLAS / OpenMP — one knob each so numpy/scipy/sklearn don't fan out.
export GOMP_SPINCOUNT=0      # prevent idle OpenMP threads from busy-waiting
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"
export NUMBA_NUM_THREADS="$THREADS"   # numba-backed losses (DTW/librosa) else fan out
export RAYON_NUM_THREADS="$THREADS"   # rust/rayon backends (e.g. some resamplers)

# JAX/XLA: disable multi-threaded Eigen (=true enables it — common footgun).
# TF inter/intra-op pools catch any remaining TF-backed threading.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export TF_NUM_INTEROP_THREADS="$THREADS"
export TF_NUM_INTRAOP_THREADS="$THREADS"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl}"
mkdir -p "$MPLCONFIGDIR" 2>/dev/null

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/numba}"
mkdir -p "$NUMBA_CACHE_DIR" 2>/dev/null

# HuggingFace cache for the CLAP loss model. The default ~/.cache/huggingface is
# read-only under the sandbox, so point HF at the repo-local writable cache.
_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export HF_HOME="${HF_HOME:-$_REPO_DIR/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HUB_CACHE" 2>/dev/null

export PY="$ENV/bin/python"
echo "[env_capped] THREADS=$THREADS  PY=$PY  (keep jobs*THREADS <= 30)"
