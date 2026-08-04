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
  TARGET="${TARGET:-disk_1111}"
fi

BASE_DIR=$(cd "$(dirname "$0")/.." && pwd)
VENV_PYTHON="$BASE_DIR/.envs/.deepinv/bin/python"
BENCH_PYTHON="$BASE_DIR/.envs/.benchmark/bin/python"

if [ -d "$BASE_DIR/data/simulated/$TARGET" ]; then
    MS_FILE="$BASE_DIR/data/simulated/$TARGET/${TARGET}.MS"
    OUTPUT_DIR="$BASE_DIR/experiments/simulated/$TARGET/${TARGET}_deepinverse"
else
    MS_FILE="$BASE_DIR/data/archival/$TARGET/${TARGET}.MS"
    OUTPUT_DIR="$BASE_DIR/experiments/archival/$TARGET/${TARGET}_deepinverse"
fi

LOG_FILE="$OUTPUT_DIR/${TARGET}_deepinverse_script.log"

YAML_PATH=""
if [ -d "$BASE_DIR/data/simulated/$TARGET" ]; then
    YAML_PATH="$BASE_DIR/data/simulated/$TARGET/${TARGET}.yaml"
else
    YAML_PATH="$BASE_DIR/data/archival/$TARGET/${TARGET}.yaml"
fi

METHOD="deepinv"
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

# Build DEEPINV_CONFIG JSON
if [ "$USE_DEFAULTS" -eq 1 ]; then
  DEEPINV_CONFIG='{
    "img_size": [64, 64],
    "real_projection": true,
    "prior": "wavelet",
    "prior_level": 3,
    "prior_wavelets": ["db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8"],
    "prior_p": 1,
    "prior_clamp_min": 0,
    "stepsize_factor": 1.5,
    "lambda_factor": 0.001,
    "params_algo_a": 3,
    "max_iter": 50,
    "early_stop": true,
    "thres_conv": 1e-5
  }'
else
  DEEPINV_CONFIG="$METHOD_CONFIG"
fi

echo "Starting DeepInv for $TARGET..."

# Generate Briggs weights
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY-RUN] ./scripts/generate_weights.sh $TARGET $NVAL"
else
    ./scripts/generate_weights.sh "$TARGET" "$NVAL"
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY-RUN] $VENV_PYTHON src/algorithms/deepinverse.py --target $TARGET --config '$DEEPINV_CONFIG' --nval $NVAL | tee $LOG_FILE"
else
    "$VENV_PYTHON" src/algorithms/deepinverse.py --target "$TARGET" --config "$DEEPINV_CONFIG" --nval "$NVAL" | tee "$LOG_FILE"
fi

# Calculate beam parameters from the PSF, then format all output FITS
PSF_FILE="$OUTPUT_DIR/${TARGET}_deepinverse_psf.fits"
if [ -f "$PSF_FILE" ]; then
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[DRY-RUN] $BENCH_PYTHON -c 'calculate_beam + formatfits on $PSF_FILE'"
    else
        echo "Calculating beam parameters from PSF..."
        "$BENCH_PYTHON" -c "
import sys, glob; sys.path.append('$BASE_DIR/src')
from utils.transforms import calculate_beam, formatfits

beam = calculate_beam('$PSF_FILE', '$MS_FILE')
print(f\"Beam: bmaj={beam['bmaj']:.4f} arcsec, bmin={beam['bmin']:.4f} arcsec, bpa={beam['bpa']:.2f} deg\")

for f in glob.glob('$OUTPUT_DIR/*.fits'):
    print(f'Formatting {f}...')
    formatfits(f, '$MS_FILE', beam_params=beam)
print('Done formatting all FITS files.')
" 2>&1 | tee -a "$LOG_FILE"
    fi
fi
