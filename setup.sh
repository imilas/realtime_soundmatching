#!/bin/bash
# One-time setup. Run from the repo root with conda activated:
#
#   conda create -n soundmatch python=3.10
#   conda activate soundmatch
#   conda install -c conda-forge faust jax jaxlib flax optax
#   bash setup.sh
#
# After this script completes, run experiments with:
#   bash experiment_scripts/run_parallel.sh

set -euo pipefail

conda install -c conda-forge faust jax jaxlib flax optax
pip install -r requirements.txt
python -c "
from synths.build import prepare
from synths.program import PROGRAMS
for name in PROGRAMS:
    print(f'  {name}...', end=' ', flush=True)
    prepare(name, force=True)
    print('ok')
"

echo ""
echo "Setup complete. Run experiments with:"
echo "  bash experiment_scripts/run_parallel.sh"
