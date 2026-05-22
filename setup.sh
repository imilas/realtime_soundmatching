#!/bin/bash
# One-time setup: install tools, create .venv, compile synths.
# No sudo required — all tools install to ~/.local or local dirs.
#
# After this script completes, run experiments with:
#   bash experiment_scripts/run_parallel.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_BIN="$HOME/.local/bin"
mkdir -p "$LOCAL_BIN"

# Add ~/.local/bin to PATH for this session
export PATH="$LOCAL_BIN:$PATH"

# ---------------------------------------------------------------------------
# 1. Python 3.10
# ---------------------------------------------------------------------------
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

echo "Using $($PYTHON --version) at $PYTHON"

# ---------------------------------------------------------------------------
# 2. Faust compiler
# ---------------------------------------------------------------------------
if ! command -v faust &>/dev/null; then
    echo "faust not found — downloading pre-built binary..."
    FAUST_VERSION="2.85.5"
    FAUST_ARCHIVE="Faust-${FAUST_VERSION}-linux-amd64.tar.gz"
    FAUST_URL="https://github.com/grame-cncm/faust/releases/download/${FAUST_VERSION}/${FAUST_ARCHIVE}"
    FAUST_DIR="$HOME/.local/faust-${FAUST_VERSION}"

    if [ ! -d "$FAUST_DIR" ]; then
        TMP=$(mktemp -d)
        curl -fsSL "$FAUST_URL" -o "$TMP/$FAUST_ARCHIVE"
        tar xf "$TMP/$FAUST_ARCHIVE" -C "$TMP"
        mv "$TMP/Faust-${FAUST_VERSION}-linux-amd64" "$FAUST_DIR"
        rm -rf "$TMP"
    fi

    # Symlink faust and faust2jaqt into ~/.local/bin
    ln -sf "$FAUST_DIR/bin/faust"      "$LOCAL_BIN/faust"
    ln -sf "$FAUST_DIR/bin/faust2jaqt" "$LOCAL_BIN/faust2jaqt"
    export PATH="$FAUST_DIR/bin:$PATH"
fi

echo "Using faust $(faust --version 2>&1 | head -1)"

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
# 4. Python venv + packages
# ---------------------------------------------------------------------------
cd "$REPO_DIR"

if [ ! -d .venv ]; then
    echo "Creating .venv..."
    "$PYTHON" -m venv .venv
fi

source .venv/bin/activate
echo "Installing Python packages..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

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
