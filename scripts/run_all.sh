#!/bin/bash
set -euo pipefail

TARGET=""
DRY_RUN=0
COMPUTE_METRICS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --compute_metrics)
      COMPUTE_METRICS=1; shift ;;
    *)
      echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [ -z "$TARGET" ]; then
  echo "Usage: $0 --target <target> [--dry-run] [--compute_metrics]"
  exit 1
fi

BASE_DIR=$(cd "$(dirname "$0")/.." && pwd)
BENCH_PYTHON="$BASE_DIR/.envs/.benchmark/bin/python"

for script in run_casa.sh run_wsclean.sh run_deepinv.sh run_r2d2.sh; do
  echo "========================================"
  echo "Running $script for $TARGET"
  echo "========================================"
  if [ "$DRY_RUN" -eq 1 ]; then
    bash "$BASE_DIR/scripts/$script" --target "$TARGET" --dry-run || echo "Warning: $script failed or was skipped."
  else
    bash "$BASE_DIR/scripts/$script" --target "$TARGET" || echo "Warning: $script failed or was skipped."
  fi
done

if [ "$COMPUTE_METRICS" -eq 1 ]; then
  echo "========================================"
  echo "Computing metrics for $TARGET"
  echo "========================================"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY-RUN] $BENCH_PYTHON src/compute_metrics.py --target $TARGET"
  else
    "$BENCH_PYTHON" src/compute_metrics.py --target "$TARGET"
  fi
fi

echo "All methods finished for $TARGET."
