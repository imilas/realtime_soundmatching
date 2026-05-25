#!/bin/bash
# One-time setup: install tools, create .venv (or use conda env), compile synths.
# No sudo required.
#
# Recommended (conda — works on servers):
#   conda create -n soundmatch python=3.10
#   conda activate soundmatch
#   conda install -c conda-forge faust jax jaxlib flax optax dm-pix
#   bash setup.sh
#
# On a local machine with sudo:
#   sudo pacman -S faust gnu-parallel python310   # Arch
#   sudo apt install faust parallel python3.10    # Ubuntu
#   bash setup.sh
#
# After this script completes, run experiments with:
#   bash experiment_scripts/run_parallel.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_BIN="$HOME/.local/bin"
mkdir -p "$LOCAL_BIN"
export PATH="$LOCAL_BIN:$PATH"

# ---------------------------------------------------------------------------
# Detect conda environment — if active, use it directly instead of a venv
# ---------------------------------------------------------------------------
IN_CONDA=false
if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" != "base" ]; then
    echo "Active conda env: $CONDA_DEFAULT_ENV — will use it instead of .venv"
    IN_CONDA=true
    PYTHON=$(command -v python)
fi

# ---------------------------------------------------------------------------
# 1. Python 3.10
# ---------------------------------------------------------------------------
if [ "$IN_CONDA" = false ]; then
    PYTHON=$(command -v python3.10 || true)

    if [ -z "$PYTHON" ]; then
        echo "python3.10 not found — installing via pyenv..."
        if [ ! -d "$HOME/.pyenv" ]; then
            curl -fsSL https://pyenv.run | bash
        fi
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init -)"
        pyenv install -s 3.10.13
        pyenv local 3.10.13
        PYTHON=$(command -v python3.10)
    fi
fi

echo "Using $($PYTHON --version) at $PYTHON"

# ---------------------------------------------------------------------------
# 2. Faust compiler
# ---------------------------------------------------------------------------
if ! command -v faust &>/dev/null; then
    echo ""
    echo "faust not found. Options:"
    echo "  1) Build from source (~10 min, needs cmake + gcc)"
    echo "  2) Skip — I will install it myself and re-run setup.sh"
    echo ""
    read -rp "Choice [1/2]: " FAUST_CHOICE

    if [ "${FAUST_CHOICE}" = "1" ]; then
        FAUST_VERSION="2.85.5"
        FAUST_SRC_URL="https://github.com/grame-cncm/faust/releases/download/${FAUST_VERSION}/faust-${FAUST_VERSION}.tar.gz"
        TMP=$(mktemp -d)
        curl -fsSL "$FAUST_SRC_URL" -o "$TMP/faust.tar.gz"
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
        cd "$REPO_DIR"
        rm -rf "$TMP"
    else
        echo ""
        echo "Install faust and re-run setup.sh. Some options:"
        echo "  sudo apt install faust"
        echo "  sudo pacman -S faust"
        echo "  conda install -c conda-forge faust"
        exit 0
    fi

    if ! command -v faust &>/dev/null; then
        echo "ERROR: faust still not found after build." >&2
        exit 1
    fi
fi

echo "Using $(faust --version 2>&1 | head -1)"

# ---------------------------------------------------------------------------
# 3. GNU parallel (optional — run_parallel.sh falls back to bash if missing)
# ---------------------------------------------------------------------------
if ! command -v parallel &>/dev/null; then
    echo "GNU parallel not found — installing to ~/.local..."
    TMP=$(mktemp -d)
    curl -fsSL https://ftpmirror.gnu.org/parallel/parallel-latest.tar.bz2 -o "$TMP/parallel.tar.bz2"
    tar xf "$TMP/parallel.tar.bz2" -C "$TMP"
    cd "$TMP"/parallel-*/
    ./configure --prefix="$HOME/.local" --quiet
    make --quiet
    make install --quiet
    cd "$REPO_DIR"
    rm -rf "$TMP"
fi

if command -v parallel &>/dev/null; then
    echo "Using $(parallel --version | head -1)"
else
    echo "GNU parallel unavailable — run_parallel.sh will use bash background jobs"
fi

# ---------------------------------------------------------------------------
# 4. Python packages
# ---------------------------------------------------------------------------
cd "$REPO_DIR"

if [ "$IN_CONDA" = true ]; then
    echo "Installing Python packages into conda env..."
    pip install -r requirements.txt
else
    if [ ! -d .venv ]; then
        echo "Creating .venv..."
        "$PYTHON" -m venv .venv
    fi
    source .venv/bin/activate
    echo "Installing Python packages..."
    pip install --upgrade pip --quiet
    pip install -r requirements.txt
fi

# ---------------------------------------------------------------------------
# 5. Compile synths (always recompile to pick up any toolchain changes)
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
