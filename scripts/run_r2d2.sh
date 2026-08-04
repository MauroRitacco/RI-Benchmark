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
R2D2_REPO="$BASE_DIR/external/R2D2-RI"
VENV_PYTHON="$BASE_DIR/.envs/.r2d2/bin/python"
BENCH_PYTHON="$BASE_DIR/.envs/.benchmark/bin/python"

if [ -d "$BASE_DIR/data/simulated/$TARGET" ]; then
    MS_FILE="$BASE_DIR/data/simulated/$TARGET/${TARGET}.MS"
    DATA_FILE="$BASE_DIR/data/simulated/$TARGET/${TARGET}.mat"
    OUTPUT_DIR="$BASE_DIR/experiments/simulated/$TARGET/${TARGET}_r2d2"
    LOG_FILE="$OUTPUT_DIR/${TARGET}_r2d2.log"
    UV_CUTOFF_FLAG=""
else
    MS_FILE="$BASE_DIR/data/archival/$TARGET/${TARGET}.MS"
    DATA_FILE="$BASE_DIR/data/archival/$TARGET/${TARGET}.mat"
    OUTPUT_DIR="$BASE_DIR/experiments/archival/$TARGET/${TARGET}_r2d2"
    LOG_FILE="$OUTPUT_DIR/${TARGET}_r2d2.log"
    UV_CUTOFF_FLAG="--uv_cutoff=2000000"
fi

YAML_PATH=""
if [ -d "$BASE_DIR/data/simulated/$TARGET" ]; then
    YAML_PATH="$BASE_DIR/data/simulated/$TARGET/${TARGET}.yaml"
else
    YAML_PATH="$BASE_DIR/data/archival/$TARGET/${TARGET}.yaml"
fi

METHOD="r2d2"
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

# Build imager.py arguments
IMAGER_CONFIG=$(yaml_get "imager_config" "config/imaging/R2D2_small.yaml")
ARCHITECTURE=$(yaml_get "architecture" "unet")
PRUNE=$(yaml_get "prune" "true")
SIGMA_RES_TOL=$(yaml_get "sigma_res_tol" "1e-4")
SUPER_RESOLUTION=$(yaml_get "super_resolution" "1")
MEAS_OP_ON_GPU=$(yaml_get "meas_op_on_gpu" "true")
IM_DIM_X=$(yaml_get "im_dim_x" "64")
IM_DIM_Y=$(yaml_get "im_dim_y" "64")
WEIGHT_ROBUSTNESS=$(yaml_get "weight_robustness" "2")
WEIGHT_TYPE=$(yaml_get "weight_type" "briggs")
NUM_ITER=$(yaml_get "num_iter" "3")
SERIES=$(yaml_get "series" "R2D2")
LAYERS=$(yaml_get "layers" "1")
CKPT_PATH=$(yaml_get "ckpt_path" "ckpt/R2D2_small_ALMA2")
CKPT_REALISATIONS=$(yaml_get "ckpt_realisations" "1")
NUM_CHANS=$(yaml_get "num_chans" "32")

# Prepend R2D2_REPO to relative paths
if [[ "$IMAGER_CONFIG" != /* ]]; then
    IMAGER_CONFIG="$R2D2_REPO/$IMAGER_CONFIG"
fi
if [[ "$CKPT_PATH" != /* ]]; then
    CKPT_PATH="$R2D2_REPO/$CKPT_PATH"
fi

# Build boolean flags
PRUNE_FLAG=""
if [ "$PRUNE" = "true" ]; then
    PRUNE_FLAG="--prune"
fi

MEAS_OP_FLAG=""
if [ "$MEAS_OP_ON_GPU" = "true" ]; then
    MEAS_OP_FLAG="--meas_op_on_gpu"
fi

# Clean up previous run outputs to ensure a fresh start
if [ -d "$OUTPUT_DIR" ]; then
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[DRY-RUN] rm -rf $OUTPUT_DIR"
    else
        rm -rf "$OUTPUT_DIR"
    fi
fi
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY-RUN] mkdir -p $OUTPUT_DIR"
else
    mkdir -p "$OUTPUT_DIR"
fi

echo "Starting R2D2 for $TARGET..."

pushd "$R2D2_REPO" > /dev/null

start_time=$(date +%s.%N)

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY-RUN] $VENV_PYTHON src/imager.py \\"
    echo "    --config $IMAGER_CONFIG \\"
    echo "    --output_path $OUTPUT_DIR \\"
    echo "    --data_file $DATA_FILE \\"
    echo "    --save_all_outputs \\"
    echo "    --architecture $ARCHITECTURE \\"
    echo "    $PRUNE_FLAG \\"
    echo "    --sigma_res_tol $SIGMA_RES_TOL \\"
    echo "    --super_resolution $SUPER_RESOLUTION \\"
    echo "    $MEAS_OP_FLAG \\"
    echo "    --im_dim_x $IM_DIM_X \\"
    echo "    --im_dim_y $IM_DIM_Y \\"
    echo "    --weight_robustness $WEIGHT_ROBUSTNESS \\"
    echo "    --weight_type $WEIGHT_TYPE \\"
    echo "    --num_iter $NUM_ITER \\"
    echo "    --series $SERIES \\"
    echo "    --layers $LAYERS \\"
    echo "    --ckpt_path $CKPT_PATH \\"
    echo "    --ckpt_realisations $CKPT_REALISATIONS \\"
    echo "    --num_chans $NUM_CHANS"
    runtime="0"
else
    "$VENV_PYTHON" src/imager.py \
        --config "$IMAGER_CONFIG" \
        --output_path "$OUTPUT_DIR" \
        --data_file "$DATA_FILE" \
        --save_all_outputs \
        --architecture "$ARCHITECTURE" \
        $PRUNE_FLAG \
        --sigma_res_tol "$SIGMA_RES_TOL" \
        --super_resolution "$SUPER_RESOLUTION" \
        $MEAS_OP_FLAG \
        --im_dim_x "$IM_DIM_X" \
        --im_dim_y "$IM_DIM_Y" \
        --weight_robustness "$WEIGHT_ROBUSTNESS" \
        --weight_type "$WEIGHT_TYPE" \
        --num_iter "$NUM_ITER" \
        --series "$SERIES" \
        --layers "$LAYERS" \
        --ckpt_path "$CKPT_PATH" \
        --ckpt_realisations "$CKPT_REALISATIONS" \
        --num_chans "$NUM_CHANS" \
        2>&1 | tee "$LOG_FILE"

    end_time=$(date +%s.%N)
    runtime=$(echo "$end_time - $start_time" | bc)
    echo -e "\nExecution time: $runtime seconds" >> "$LOG_FILE"
fi

popd > /dev/null

# Rename the output FITS files and move other outputs one directory up
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY-RUN] mv $OUTPUT_DIR/$TARGET/R2D2_model_image.fits $OUTPUT_DIR/${TARGET}_r2d2.fits (if exists)"
    echo "[DRY-RUN] mv $OUTPUT_DIR/$TARGET/PSF.fits $OUTPUT_DIR/${TARGET}_r2d2_psf.fits (if exists)"
    echo "[DRY-RUN] mv $OUTPUT_DIR/$TARGET/* $OUTPUT_DIR/ (if dir exists)"
    echo "[DRY-RUN] rmdir $OUTPUT_DIR/$TARGET (if empty)"
else
    if [ -d "$OUTPUT_DIR/$TARGET" ]; then
        if [ -f "$OUTPUT_DIR/$TARGET/R2D2_model_image.fits" ]; then
            mv "$OUTPUT_DIR/$TARGET/R2D2_model_image.fits" "$OUTPUT_DIR/${TARGET}_r2d2.fits"
        fi
        if [ -f "$OUTPUT_DIR/$TARGET/PSF.fits" ]; then
            mv "$OUTPUT_DIR/$TARGET/PSF.fits" "$OUTPUT_DIR/${TARGET}_r2d2_psf.fits"
        fi
        mv "$OUTPUT_DIR/$TARGET"/* "$OUTPUT_DIR/" 2>/dev/null || true
        rmdir "$OUTPUT_DIR/$TARGET" 2>/dev/null || true
    fi
fi

# Calculate beam parameters from the PSF, then format all output FITS
PSF_FILE="$OUTPUT_DIR/${TARGET}_r2d2_psf.fits"
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
