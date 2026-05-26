#!/bin/bash
# One-time setup. Run from the repo root with conda activated:
#
#   conda create -n soundmatch python=3.10
#   conda activate soundmatch
#   bash setup.sh
#
# Faust will be built from source if not already installed or if the installed
# compiler cannot emit C++ code, which faust2sndfile requires.
# After this script completes, run experiments with:
#   bash experiment_scripts/run_parallel.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------------------
# 1. Faust
# ---------------------------------------------------------------------------
faust_has_cpp_backend() {
    local check_dir check_dsp
    check_dir="$(mktemp -d)"
    check_dsp="$check_dir/check.dsp"
    printf 'process = _;\n' > "$check_dsp"
    if faust -lang cpp "$check_dsp" -o "$check_dir/check.cpp" >/dev/null 2>&1; then
        rm -rf "$check_dir"
        return 0
    fi
    rm -rf "$check_dir"
    return 1
}

if ! command -v faust &>/dev/null || ! faust_has_cpp_backend; then
    echo "faust not found or missing C++ backend — building from source (~10 min)..."
    FAUST_VERSION="2.85.5"
    TMP=$(mktemp -d)
    curl -fsSL "https://github.com/grame-cncm/faust/releases/download/${FAUST_VERSION}/faust-${FAUST_VERSION}.tar.gz" -o "$TMP/faust.tar.gz"
    tar xf "$TMP/faust.tar.gz" -C "$TMP"
    cd "$TMP"/faust-*
    make compiler -j"$(nproc)" PREFIX="$HOME/.local"
    make install PREFIX="$HOME/.local"
    export PATH="$HOME/.local/bin:$PATH"
    cd "$REPO_DIR"
    rm -rf "$TMP"
fi

if ! faust_has_cpp_backend; then
    echo "faust is installed, but it still cannot emit C++ code." >&2
    echo "Check that $HOME/.local/bin is before any broken faust installation in PATH." >&2
    exit 1
fi

echo "Using $(faust --version 2>&1 | head -1)"

# ---------------------------------------------------------------------------
# 2. JAX stack and native audio build deps (via conda)
# ---------------------------------------------------------------------------
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    cat >&2 <<'EOF'
No conda environment is active.

Create and activate the project environment first:
  conda create -n soundmatch python=3.10
  conda activate soundmatch
  bash setup.sh

Or run directly with:
  conda run -n soundmatch bash setup.sh
EOF
    exit 1
fi

MISSING_CONDA_PACKAGES=()

python - <<'EOF' || MISSING_CONDA_PACKAGES+=(jax jaxlib flax optax)
import importlib.util
import sys

missing = [
    pkg
    for pkg in ("jax", "jaxlib", "flax", "optax")
    if importlib.util.find_spec(pkg) is None
]
if missing:
    print("Missing conda packages:", ", ".join(missing))
    sys.exit(1)

print("JAX stack already installed")
EOF

if ! command -v pkg-config >/dev/null 2>&1 || ! pkg-config --exists sndfile; then
    MISSING_CONDA_PACKAGES+=(libsndfile pkg-config)
else
    echo "Native libsndfile build deps already installed"
fi

if ((${#MISSING_CONDA_PACKAGES[@]})); then
    conda install -y -c conda-forge "${MISSING_CONDA_PACKAGES[@]}"
fi

# ---------------------------------------------------------------------------
# 3. Python packages (via pip)
# ---------------------------------------------------------------------------
python -m pip install -r requirements.txt

# ---------------------------------------------------------------------------
# 4. Compile synths
# ---------------------------------------------------------------------------
echo "Building synths..."
python - <<'EOF'
from synths.build import prepare
from synths.program import PROGRAMS
for name in PROGRAMS:
    print(f"  {name}...", end=" ", flush=True)
    prepare(name, force=True)
    print("ok")
EOF

echo ""
echo "Setup complete. Run experiments with:"
echo "  bash experiment_scripts/run_parallel.sh"
