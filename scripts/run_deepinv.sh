#!/bin/bash
TARGET=disk_0005
N=3

BASE_DIR=$(pwd)
VENV_PYTHON="$BASE_DIR/.envs/.deepinv/bin/python"

echo "Starting DeepInv for $TARGET..."

# Generate Briggs weights
./scripts/generate_weights.sh "$TARGET" "$N"

"$VENV_PYTHON" src/algorithms/deepinverse.py --target "$TARGET"

