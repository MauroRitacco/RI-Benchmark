#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENVS_DIR="$PROJECT_DIR/.envs"
mkdir -p "$ENVS_DIR"

ENV_NAMES=(".casa" ".deepinv" ".r2d2" ".wsclean" ".benchmark")

echo "Creating venvs in $ENVS_DIR/..."

for ENV_NAME in "${ENV_NAMES[@]}"; do
    ENV_PATH="$ENVS_DIR/$ENV_NAME"
    
    if [ -d "$ENV_PATH" ]; then
        echo "[SKIP] $ENV_NAME already exists."
    else
        echo "[CREATING] $ENV_NAME..."
        python3.10 -m venv "$ENV_PATH"
        echo "[DONE] $ENV_NAME created."
    fi
done

echo "Finished."
