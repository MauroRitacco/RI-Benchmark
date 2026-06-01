#!/bin/bash
TARGET=disk_0005

BASE_DIR=$(pwd)
VENV_PYTHON="$BASE_DIR/.envs/.casa/bin/python"

echo "Starting casa for $TARGET..."

"$VENV_PYTHON" src/algorithms/casa.py --target "$TARGET"

