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
WSCLEAN_VENV="$BASE_DIR/.envs/.wsclean/bin/activate"
BENCH_PYTHON="$BASE_DIR/.envs/.benchmark/bin/python"

if [ -d "$BASE_DIR/data/simulated/$TARGET" ]; then
    DATA_FILE="$BASE_DIR/data/simulated/$TARGET/${TARGET}.MS"
    OUTPUT_DIR="$BASE_DIR/experiments/simulated/$TARGET/${TARGET}_wsclean"
    LOG_FILE="$OUTPUT_DIR/${TARGET}_wsclean.log"
else
    DATA_FILE="$BASE_DIR/data/archival/$TARGET/${TARGET}.MS"
    OUTPUT_DIR="$BASE_DIR/experiments/archival/$TARGET/${TARGET}_wsclean"
    LOG_FILE="$OUTPUT_DIR/${TARGET}_wsclean.log"
fi

YAML_PATH=""
if [ -d "$BASE_DIR/data/simulated/$TARGET" ]; then
    YAML_PATH="$BASE_DIR/data/simulated/$TARGET/${TARGET}.yaml"
else
    YAML_PATH="$BASE_DIR/data/archival/$TARGET/${TARGET}.yaml"
fi

METHOD="wsclean"
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

# Helper to extract a single key from YAML or fallback to default
yaml_get() {
  local key="$1"
  local default_val="$2"
  if [ "$USE_DEFAULTS" -eq 1 ]; then
    echo "$default_val"
    return
  fi
  local val
  val=$("$BENCH_PYTHON" "$BASE_DIR/scripts/parse_target_yaml.py" \
    --target "$TARGET" --method "$METHOD" --key "$key" 2>/dev/null) || true
  if [ -z "$val" ]; then
    echo "$default_val"
  else
    echo "$val"
  fi
}

# Build wsclean config
SIZE=$(yaml_get "size" "64")
PADDING=$(yaml_get "padding" "2")
J=$(yaml_get "j" "1")
WEIGHT=$(yaml_get "weight" "briggs 2")
MULTISCALE=$(yaml_get "multiscale" "true")
MULTISCALE_SCALES=$(yaml_get "multiscale_scales" "0,5,15")
MGAIN=$(yaml_get "mgain" "0.8")
GAIN=$(yaml_get "gain" "0.1")
AUTO_THRESHOLD=$(yaml_get "auto_threshold" "0.5")
AUTO_MASK=$(yaml_get "auto_mask" "1.5")
NITER=$(yaml_get "niter" "200000")
DATA_COLUMN=$(yaml_get "data_column" "DATA")

# Load pixel size from .mat file
MAT_FILE="${DATA_FILE%.MS}.mat"
PIXEL_SIZE=$($BENCH_PYTHON -c "
from scipy.io import loadmat
pixel_size = float(loadmat('$MAT_FILE')['nominal_pixelsize'].item())
print(f'{pixel_size}asec')
")

# Eliminate the directory to ensure a fresh run
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY-RUN] rm -rf $OUTPUT_DIR"
    echo "[DRY-RUN] mkdir -p $OUTPUT_DIR"
    echo "[DRY-RUN] cd $OUTPUT_DIR"
else
    rm -rf "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
    cd "$OUTPUT_DIR"
fi

echo "Starting WSClean for $TARGET..."
echo "Log saved in: $LOG_FILE"

# Build dynamic multiscale flags
MULTISCALE_FLAGS=""
if [ "$MULTISCALE" = "true" ]; then
    MULTISCALE_FLAGS="-multiscale -multiscale-scales $MULTISCALE_SCALES"
fi

start_time=$(date +%s.%N)

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY-RUN] source $WSCLEAN_VENV"
    echo "[DRY-RUN] wsclean -size $SIZE $SIZE -scale $PIXEL_SIZE -padding $PADDING -j $J -weight $WEIGHT $MULTISCALE_FLAGS -mgain $MGAIN -gain $GAIN -auto-threshold $AUTO_THRESHOLD -auto-mask $AUTO_MASK -niter $NITER -data-column $DATA_COLUMN -name $TARGET $DATA_FILE"
    echo "[DRY-RUN] mv ${TARGET}-image.fits ${TARGET}_wsclean.fits"
    echo "[DRY-RUN] $BENCH_PYTHON -c 'convert_jybeam_to_jypixel on ${TARGET}_wsclean.fits'"
    runtime="0"
else
    source "$WSCLEAN_VENV"

    wsclean -size "$SIZE" "$SIZE" \
        -scale "$PIXEL_SIZE" \
        -padding "$PADDING" \
        -j "$J" \
        -weight $WEIGHT \
        $MULTISCALE_FLAGS \
        -mgain "$MGAIN" \
        -gain "$GAIN" \
        -auto-threshold "$AUTO_THRESHOLD" \
        -auto-mask "$AUTO_MASK" \
        -niter "$NITER" \
        -data-column "$DATA_COLUMN" \
        -name "$TARGET" \
        "$DATA_FILE" 2>&1 | tee "$LOG_FILE"

    mv "${TARGET}-image.fits" "${TARGET}_wsclean.fits"
    "$BENCH_PYTHON" -c "import sys; sys.path.append('$BASE_DIR/src'); from utils.transforms import convert_jybeam_to_jypixel; convert_jybeam_to_jypixel('$OUTPUT_DIR/${TARGET}_wsclean.fits')"

    end_time=$(date +%s.%N)
    runtime=$(echo "$end_time - $start_time" | bc)

    echo "wsclean config: -size $SIZE $SIZE -scale $PIXEL_SIZE -padding $PADDING -j $J -weight $WEIGHT $MULTISCALE_FLAGS -mgain $MGAIN -gain $GAIN -auto-threshold $AUTO_THRESHOLD -auto-mask $AUTO_MASK -niter $NITER -data-column $DATA_COLUMN" >> "$LOG_FILE"
    echo "Execution time: $runtime seconds" >> "$LOG_FILE"
fi
