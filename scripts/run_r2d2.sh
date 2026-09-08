#!/bin/bash
set -euo pipefail

TARGET=""
DRY_RUN=0
OUTPUT_DIR=""
CONFIG_JSON=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --config)
      CONFIG_JSON="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [ -z "$TARGET" ]; then
  echo "Usage: $0 --target <target> [--dry-run] [--output-dir DIR] [--config JSON]"
  exit 1
fi

BASE_DIR=$(cd "$(dirname "$0")/.." && pwd)
R2D2_REPO="$BASE_DIR/external/R2D2-RI"
VENV_PYTHON="$BASE_DIR/.envs/.r2d2/bin/python"
BENCH_PYTHON="$BASE_DIR/.envs/.benchmark/bin/python"

if [ "${RI_BENCH_TEST:-0}" = "1" ]; then
    SIM_DIR="$BASE_DIR/data/simulated/test"
else
    SIM_DIR="$BASE_DIR/data/simulated"
fi

if [ -d "$SIM_DIR/$TARGET" ]; then
    MS_FILE="$SIM_DIR/$TARGET/${TARGET}.MS"
    DATA_FILE="$SIM_DIR/$TARGET/${TARGET}.mat"
    if [ "${RI_BENCH_TEST:-0}" = "1" ]; then
        CATEGORY="test"
    else
        CATEGORY="simulated"
    fi
    UV_CUTOFF_FLAG=""
else
    MS_FILE="$BASE_DIR/data/archival/$TARGET/${TARGET}.MS"
    DATA_FILE="$BASE_DIR/data/archival/$TARGET/${TARGET}.mat"
    CATEGORY="archival"
    UV_CUTOFF_FLAG="--uv_cutoff=2000000"
fi

# Default: standard benchmark layout; --output-dir overrides.
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$BASE_DIR/experiments/$CATEGORY/$TARGET/${TARGET}_r2d2"
fi
LOG_FILE="$OUTPUT_DIR/${TARGET}_r2d2.log"

YAML_PATH=""
if [ -d "$SIM_DIR/$TARGET" ]; then
    YAML_PATH="$SIM_DIR/$TARGET/${TARGET}.yaml"
else
    YAML_PATH="$BASE_DIR/data/archival/$TARGET/${TARGET}.yaml"
fi

METHOD="r2d2"
METHOD_CONFIG=""
if [ -f "$YAML_PATH" ]; then
  if [ -n "$CONFIG_JSON" ]; then
    METHOD_CONFIG=$("$BENCH_PYTHON" "$BASE_DIR/scripts/parse_target_yaml.py" \
      --target "$TARGET" --method "$METHOD" --config "$CONFIG_JSON" 2>/dev/null) || true
  else
    METHOD_CONFIG=$("$BENCH_PYTHON" "$BASE_DIR/scripts/parse_target_yaml.py" \
      --target "$TARGET" --method "$METHOD" 2>/dev/null) || true
  fi
fi

USE_DEFAULTS=0
if [ -z "$METHOD_CONFIG" ] || [ "$METHOD_CONFIG" = "null" ]; then
  if [ -n "$CONFIG_JSON" ]; then
    # Explicit JSON config: use it even if target YAML has no method section.
    METHOD_CONFIG=$("$BENCH_PYTHON" -c "import json,sys; print(json.dumps(json.loads(sys.argv[1])))" "$CONFIG_JSON" 2>/dev/null) || true
  fi
  if [ -z "$METHOD_CONFIG" ] || [ "$METHOD_CONFIG" = "null" ]; then
    if [ -f "$YAML_PATH" ]; then
      echo "Warning: $METHOD section missing from $YAML_PATH; using hardcoded defaults." >&2
    fi
    USE_DEFAULTS=1
  fi
fi

# Get a YAML key (yaml_get key default), falling back to the default value
yaml_get() {
  local key="$1"
  local default_val="$2"
  if [ "$USE_DEFAULTS" -eq 1 ]; then
    echo "$default_val"
    return
  fi
  local val
  if [ -n "$CONFIG_JSON" ]; then
    val=$("$BENCH_PYTHON" "$BASE_DIR/scripts/parse_target_yaml.py" \
      --target "$TARGET" --method "$METHOD" --key "$key" --config "$CONFIG_JSON" 2>/dev/null) || true
  else
    val=$("$BENCH_PYTHON" "$BASE_DIR/scripts/parse_target_yaml.py" \
      --target "$TARGET" --method "$METHOD" --key "$key" 2>/dev/null) || true
  fi
  if [ -z "$val" ]; then
    echo "$default_val"
  else
    echo "$val"
  fi
}

IMAGER_CONFIG=$(yaml_get "imager_config" "$BASE_DIR/config/imaging/R2D2_small.yaml")
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

# Resolve relative paths: configs live in this repo, checkpoints in external/R2D2-RI
if [[ "$IMAGER_CONFIG" != /* ]]; then
    IMAGER_CONFIG="$BASE_DIR/$IMAGER_CONFIG"
fi
if [[ "$CKPT_PATH" != /* ]]; then
    CKPT_PATH="$R2D2_REPO/$CKPT_PATH"
fi

PRUNE_FLAG=""
if [ "$PRUNE" = "true" ]; then
    PRUNE_FLAG="--prune"
fi

MEAS_OP_FLAG=""
if [ "$MEAS_OP_ON_GPU" = "true" ]; then
    MEAS_OP_FLAG="--meas_op_on_gpu"
fi

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
