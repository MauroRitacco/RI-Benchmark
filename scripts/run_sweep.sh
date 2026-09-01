#!/usr/bin/env bash
# Run the hyper-parameter sweep over the targets listed in scripts/sweep_targets.txt.
# Usage: ./scripts/run_sweep.sh [--dry-run] [--evaluate] [--cpu-workers N] [--gpu-workers M] [extra args]
set -euo pipefail

BASE_DIR=$(cd "$(dirname "$0")/.." && pwd)
BENCH_PYTHON="$BASE_DIR/.envs/.benchmark/bin/python"
SWEEP_SCRIPT="$BASE_DIR/src/sweep/run_sweep.py"

exec "$BENCH_PYTHON" "$SWEEP_SCRIPT" "$@"
