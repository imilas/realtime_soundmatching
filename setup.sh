#!/bin/bash
# One-time setup. Run from the repo root with conda activated:
#
#   conda create -n soundmatch python=3.10
#   conda activate soundmatch
#   bash setup.sh
#
# Faust will be built from source if not already installed.
# After this script completes, run experiments with:
#   bash experiment_scripts/run_parallel.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------------------
# 1. Faust
# ---------------------------------------------------------------------------
if ! command -v faust &>/dev/null; then
    echo "faust not found — building from source (~10 min)..."
    FAUST_VERSION="2.85.5"
    TMP=$(mktemp -d)
    curl -fsSL "https://github.com/grame-cncm/faust/releases/download/${FAUST_VERSION}/faust-${FAUST_VERSION}.tar.gz" -o "$TMP/faust.tar.gz"
    tar xf "$TMP/faust.tar.gz" -C "$TMP"
    cd "$TMP"/faust-*/build
    cmake . \
        -DCMAKE_INSTALL_PREFIX="$HOME/.local" \
        -DCMAKE_BUILD_TYPE=Release \
        -DINCLUDE_LLVM=OFF \
        -DINCLUDE_OSC=ON \
        -DINCLUDE_HTTP=OFF \
        -DUSE_LLVM_CONFIG=OFF
    cmake --build . --parallel "$(nproc)" 2>&1 | grep --line-buffered -E "^\["
    cmake --install . --prefix "$HOME/.local"
    export PATH="$HOME/.local/bin:$PATH"
    cd "$REPO_DIR"
    rm -rf "$TMP"
fi

echo "Using $(faust --version 2>&1 | head -1)"

# ---------------------------------------------------------------------------
# 2. JAX stack (via conda)
# ---------------------------------------------------------------------------
conda install -y -c conda-forge jax jaxlib flax optax

# ---------------------------------------------------------------------------
# 3. Python packages (via pip)
# ---------------------------------------------------------------------------
pip install -r requirements.txt

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
