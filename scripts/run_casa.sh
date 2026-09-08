#!/bin/bash
set -euo pipefail

TARGET=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    *)
      echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [ -z "$TARGET" ]; then
  echo "Usage: $0 --target <target> [--dry-run]"
  exit 1
fi

BASE_DIR=$(cd "$(dirname "$0")/.." && pwd)
VENV_PYTHON="$BASE_DIR/.envs/.casa/bin/python"
BENCH_PYTHON="$BASE_DIR/.envs/.benchmark/bin/python"

YAML_PATH=""
if [ -d "$BASE_DIR/data/simulated/$TARGET" ]; then
    YAML_PATH="$BASE_DIR/data/simulated/$TARGET/${TARGET}.yaml"
else
    YAML_PATH="$BASE_DIR/data/archival/$TARGET/${TARGET}.yaml"
fi

METHOD="casa"
METHOD_CONFIG=""
if [ -f "$YAML_PATH" ]; then
  METHOD_CONFIG=$("$BENCH_PYTHON" "$BASE_DIR/scripts/parse_target_yaml.py" \
    --target "$TARGET" --method "$METHOD" 2>/dev/null) || true
fi

USE_DEFAULTS=0
if [ -z "$METHOD_CONFIG" ] || [ "$METHOD_CONFIG" = "null" ]; then
  if [ -f "$YAML_PATH" ]; then
    echo "Warning: $METHOD section missing from $YAML_PATH; using hardcoded defaults." >&2
  fi
  USE_DEFAULTS=1
fi

# Read nval from top-level YAML or default to 1
NVAL="1"
if [ -f "$YAML_PATH" ]; then
  NVAL=$("$BENCH_PYTHON" "$BASE_DIR/scripts/parse_target_yaml.py" \
    --target "$TARGET" --key nval 2>/dev/null) || true
fi
if [ -z "$NVAL" ]; then
  NVAL="1"
fi

if [ "$USE_DEFAULTS" -eq 1 ]; then
  TCLEAN_CONFIG='{
    "imsize": 64,
    "specmode": "mfs",
    "deconvolver": "hogbom",
    "gridder": "standard",
    "weighting": "briggs",
    "robust": 2,
    "gain": 0.1,
    "niter": 1000000000,
    "threshold": "0.05Jy",
    "datacolumn": "data"
  }'
else
  TCLEAN_CONFIG="$METHOD_CONFIG"
fi

echo "Starting casa for $TARGET..."

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[DRY-RUN] $VENV_PYTHON src/algorithms/casa.py --target $TARGET --config '$TCLEAN_CONFIG' --nval $NVAL"
else
  "$VENV_PYTHON" src/algorithms/casa.py --target "$TARGET" --config "$TCLEAN_CONFIG" --nval "$NVAL"
fi
