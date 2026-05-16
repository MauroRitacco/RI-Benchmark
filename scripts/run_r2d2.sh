#!/bin/bash

TARGET=disk_0000

BASE_DIR=$(pwd)
R2D2_REPO="$BASE_DIR/external/R2D2-RI"
VENV_PYTHON="$BASE_DIR/.envs/.r2d2/bin/python"

if [ -d "$BASE_DIR/data/simulated/$TARGET" ]; then
    DATA_FILE="$BASE_DIR/data/simulated/$TARGET/${TARGET}.mat"
    OUTPUT_DIR="$BASE_DIR/experiments/simulated/$TARGET/${TARGET}_r2d2"
    LOG_FILE="$OUTPUT_DIR/${TARGET}_r2d2.log"
else
    DATA_FILE="$BASE_DIR/data/archival/$TARGET/${TARGET}.mat"
    OUTPUT_DIR="$BASE_DIR/experiments/archival/$TARGET/${TARGET}_r2d2"
    LOG_FILE="$OUTPUT_DIR/${TARGET}_r2d2.log"
fi

mkdir -p "$OUTPUT_DIR"

echo "Starting R2D2 for $TARGET..."
echo "Log saved in: $LOG_FILE"

pushd "$R2D2_REPO" > /dev/null

start_time=$(date +%s.%N)

"$VENV_PYTHON" src/imager.py \
    --config "$R2D2_REPO/config/imaging/R2D2_small.yaml" \
    --output_path "$OUTPUT_DIR" \
    --data_file "$DATA_FILE" \
    --save_all_outputs \
    --architecture unet \
    --prune \
    --sigma_res_tol 1e-4 \
    --super_resolution 1 \
    --meas_op_on_gpu \
    --im_dim_x 64 \
    --im_dim_y 64 \
    --weight_type briggs \
    --num_iter 3 \
    --series R2D2 \
    --layers 1 \
    --ckpt_path "$R2D2_REPO/ckpt/R2D2_small" \
    --ckpt_realisations 1 2>&1 | tee "$LOG_FILE"

end_time=$(date +%s.%N)
runtime=$(echo "$end_time - $start_time" | bc)

popd > /dev/null

echo -e "\nExecution time: $runtime seconds" >> "$LOG_FILE"

# Rename the output FITS file and move other outputs one directory up
if [ -d "$OUTPUT_DIR/$TARGET" ]; then
    if [ -f "$OUTPUT_DIR/$TARGET/R2D2_model_image.fits" ]; then
        mv "$OUTPUT_DIR/$TARGET/R2D2_model_image.fits" "$OUTPUT_DIR/${TARGET}_r2d2.fits"
    fi
    # Move remaining files up one level
    mv "$OUTPUT_DIR/$TARGET"/* "$OUTPUT_DIR/" 2>/dev/null || true
    # Remove the now empty target directory
    rmdir "$OUTPUT_DIR/$TARGET" 2>/dev/null || true
fi

echo "✅ Proceso terminado. Tiempo total: $runtime seg."