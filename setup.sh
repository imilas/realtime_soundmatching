#!/bin/bash
# One-time setup: install tools, create .venv (or use conda env), compile synths.
# No sudo required.
#
# Recommended on servers with conda:
#   conda create -n soundmatch python=3.10
#   conda activate soundmatch
#   conda install -c conda-forge faust
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
    echo "faust not found — trying to install without sudo..."

    CONDA_CMD=$(command -v mamba || command -v conda || true)
    if [ -n "$CONDA_CMD" ]; then
        echo "  using $CONDA_CMD to install faust..."
        "$CONDA_CMD" install -y -c conda-forge faust
    else
        echo "  conda/mamba not found — building faust from source (~10 min)..."
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
    fi

    if ! command -v faust &>/dev/null; then
        echo ""
        echo "ERROR: could not install faust automatically." >&2
        echo "Run: conda install -c conda-forge faust" >&2
        exit 1
    fi
fi

echo "Using $(faust --version 2>&1 | head -1)"

# ---------------------------------------------------------------------------
# 3. GNU parallel
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

echo "Using $(parallel --version | head -1)"

# ---------------------------------------------------------------------------
# 4. Python packages
# ---------------------------------------------------------------------------
cd "$REPO_DIR"

if [ "$IN_CONDA" = true ]; then
    echo "Installing Python packages into conda env..."
    pip install -r requirements.txt --quiet
else
    if [ ! -d .venv ]; then
        echo "Creating .venv..."
        "$PYTHON" -m venv .venv
    fi
    source .venv/bin/activate
    echo "Installing Python packages..."
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
fi

# ---------------------------------------------------------------------------
# 5. Compile synths (idempotent — skips if already built)
# ---------------------------------------------------------------------------
echo "Building synths..."
python - <<'EOF'
from synths.build import prepare
from synths.program import PROGRAMS
for name in PROGRAMS:
    print(f"  {name}...", end=" ", flush=True)
    prepare(name)
    print("ok")
EOF

echo ""
echo "Setup complete. Run experiments with:"
echo "  bash experiment_scripts/run_parallel.sh"
