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

SIZE=$(yaml_get "size" "64")
PADDING=$(yaml_get "padding" "2")
J=$(yaml_get "j" "1")
WEIGHT=$(yaml_get "weight" "briggs 2")
MULTISCALE=$(yaml_get "multiscale" "true")
MULTISCALE_SCALES=$(yaml_get "multiscale_scales" "0,5,15")
MGAIN=$(yaml_get "mgain" "0.8")
GAIN=$(yaml_get "gain" "0.1")
AUTO_THRESHOLD=$(yaml_get "auto_threshold" "0.5")
ABS_THRESHOLD=$(yaml_get "abs_threshold" "")
AUTO_MASK=$(yaml_get "auto_mask" "")
NITER=$(yaml_get "niter" "200000")
DATA_COLUMN=$(yaml_get "data_column" "DATA")

MAT_FILE="${DATA_FILE%.MS}.mat"
PIXEL_SIZE=$($BENCH_PYTHON -c "
from scipy.io import loadmat
pixel_size = float(loadmat('$MAT_FILE')['nominal_pixelsize'].item())
print(f'{pixel_size}asec')
")
PIXEL_SIZE_VALUE=$(echo "$PIXEL_SIZE" | sed 's/asec//')

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

MULTISCALE_FLAGS=""
if [ "$MULTISCALE" = "true" ]; then
    MULTISCALE_FLAGS="-multiscale -multiscale-scales $MULTISCALE_SCALES"
fi

AUTO_MASK_FLAGS=""
if [ -n "$AUTO_MASK" ] && [ "$AUTO_MASK" != "null" ]; then
    AUTO_MASK_FLAGS="-auto-mask $AUTO_MASK"
fi

THRESHOLD_FLAGS=""
AUTO_MULTIPLIER=""
if [ -n "$ABS_THRESHOLD" ] && [ "$ABS_THRESHOLD" != "null" ]; then
    case "$ABS_THRESHOLD" in
        auto:*)
            AUTO_MULTIPLIER="${ABS_THRESHOLD#auto:}"
            ;;
        *)
            THRESHOLD_FLAGS="-abs-threshold $ABS_THRESHOLD"
            ;;
    esac
else
    THRESHOLD_FLAGS="-auto-threshold $AUTO_THRESHOLD"
fi

# If an auto threshold was requested, run a WSClean dirty pass and compute sigma.
if [ -n "$AUTO_MULTIPLIER" ]; then
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[DRY-RUN] auto threshold: would run wsclean -niter 0 and compute sigma * $AUTO_MULTIPLIER"
        THRESHOLD_FLAGS="-abs-threshold <AUTO>"
    else
        # Ground-truth FITS for point-source filtering.
        GT_PATH=""
        if [ -f "$BASE_DIR/data/simulated/$TARGET/${TARGET}.fits" ]; then
            GT_PATH="$BASE_DIR/data/simulated/$TARGET/${TARGET}.fits"
        elif [ -f "$BASE_DIR/data/archival/$TARGET/${TARGET}.fits" ]; then
            GT_PATH="$BASE_DIR/data/archival/$TARGET/${TARGET}.fits"
        fi
        gt_arg=""
        if [ -n "$GT_PATH" ]; then
            gt_arg="--gt-path $GT_PATH"
        fi

        # Same cache as the sweep driver so standalone and sweep runs reuse noise estimates.
        WSCLEAN_DIRTY_PREFIX="$BASE_DIR/experiments/simulated/$TARGET/${TARGET}_wsclean_dirty_pass"
        WSCLEAN_DIRTY_CACHE="$BASE_DIR/experiments/simulated/$TARGET/.wsclean_dirty_noise.json"

        noise_result=$($BENCH_PYTHON "$BASE_DIR/src/utils/wsclean_dirty_noise.py" \
            --ms "$DATA_FILE" \
            --size "$SIZE" \
            --pixel-size "$PIXEL_SIZE_VALUE" \
            --weight "$WEIGHT" \
            --data-column "$DATA_COLUMN" \
            $gt_arg \
            --dirty-prefix "$WSCLEAN_DIRTY_PREFIX" \
            --cache "$WSCLEAN_DIRTY_CACHE" \
            --multiplier "$AUTO_MULTIPLIER" \
            --log "$LOG_FILE")

        eval "$noise_result"
        if [ -z "${threshold:-}" ]; then
            echo "Error: failed to compute auto threshold for $TARGET" >&2
            exit 1
        fi
        THRESHOLD_FLAGS="-abs-threshold $threshold"
    fi
fi

start_time=$(date +%s.%N)

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY-RUN] source $WSCLEAN_VENV"
    echo "[DRY-RUN] wsclean -size $SIZE $SIZE -scale $PIXEL_SIZE -padding $PADDING -gridder wstacking -wstack-nwlayers 1 -j $J -weight $WEIGHT $MULTISCALE_FLAGS -mgain $MGAIN -gain $GAIN $THRESHOLD_FLAGS $AUTO_MASK_FLAGS -niter $NITER -data-column $DATA_COLUMN -name $TARGET $DATA_FILE"
    echo "[DRY-RUN] mv ${TARGET}-image.fits ${TARGET}_wsclean.fits"
    echo "[DRY-RUN] $BENCH_PYTHON -c 'convert_jybeam_to_jypixel on ${TARGET}_wsclean.fits'"
    runtime="0"
else
    source "$WSCLEAN_VENV"

    wsclean -size "$SIZE" "$SIZE" \
        -scale "$PIXEL_SIZE" \
        -padding "$PADDING" \
        -gridder wstacking \
        -wstack-nwlayers 1 \
        -j "$J" \
        -weight $WEIGHT \
        $MULTISCALE_FLAGS \
        -mgain "$MGAIN" \
        -gain "$GAIN" \
        $THRESHOLD_FLAGS \
        $AUTO_MASK_FLAGS \
        -niter "$NITER" \
        -data-column "$DATA_COLUMN" \
        -name "$TARGET" \
        "$DATA_FILE" 2>&1 | tee "$LOG_FILE"

    mv "${TARGET}-image.fits" "${TARGET}_wsclean.fits"
    "$BENCH_PYTHON" -c "import sys; sys.path.append('$BASE_DIR/src'); from utils.transforms import convert_jybeam_to_jypixel; convert_jybeam_to_jypixel('$OUTPUT_DIR/${TARGET}_wsclean.fits')"

    end_time=$(date +%s.%N)
    runtime=$(echo "$end_time - $start_time" | bc)

    echo "wsclean config: -size $SIZE $SIZE -scale $PIXEL_SIZE -padding $PADDING -j $J -weight $WEIGHT $MULTISCALE_FLAGS -mgain $MGAIN -gain $GAIN $THRESHOLD_FLAGS $AUTO_MASK_FLAGS -niter $NITER -data-column $DATA_COLUMN" >> "$LOG_FILE"
    echo "Execution time: $runtime seconds" >> "$LOG_FILE"
fi
