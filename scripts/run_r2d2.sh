#!/bin/bash

TARGET=disk_1111

BASE_DIR=$(pwd)
R2D2_REPO="$BASE_DIR/external/R2D2-RI"
VENV_PYTHON="$BASE_DIR/.envs/.r2d2/bin/python"

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

# # Generate .mat file from .MS file
# echo "Generating .mat file for $TARGET..."
# "$BASE_DIR/.envs/.benchmark/bin/python" "$BASE_DIR/src/utils/ms2mat.py" \
#     --msfile="$MS_FILE" \
#     --srcname="$TARGET" \
#     --srcid=0 \
#     --freqid=0 \
#     $UV_CUTOFF_FLAG


# Clean up previous run outputs to ensure a fresh start
if [ -d "$OUTPUT_DIR" ]; then
    rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"

echo "Starting R2D2 for $TARGET..."

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
    --weight_robustness 2 \
    --weight_type briggs \
    --num_iter 3 \
    --series R2D2 \
    --layers 1 \
    --ckpt_path "$R2D2_REPO/ckpt/R2D2_small_ALMA2" \
    --ckpt_realisations 1 2>&1 | tee "$LOG_FILE"

end_time=$(date +%s.%N)
runtime=$(echo "$end_time - $start_time" | bc)

popd > /dev/null

echo -e "\nExecution time: $runtime seconds" >> "$LOG_FILE"

# Rename the output FITS files and move other outputs one directory up
if [ -d "$OUTPUT_DIR/$TARGET" ]; then
    if [ -f "$OUTPUT_DIR/$TARGET/R2D2_model_image.fits" ]; then
        mv "$OUTPUT_DIR/$TARGET/R2D2_model_image.fits" "$OUTPUT_DIR/${TARGET}_r2d2.fits"
    fi
    if [ -f "$OUTPUT_DIR/$TARGET/PSF.fits" ]; then
        mv "$OUTPUT_DIR/$TARGET/PSF.fits" "$OUTPUT_DIR/${TARGET}_r2d2_psf.fits"
    fi
    # Move remaining files up one level
    mv "$OUTPUT_DIR/$TARGET"/* "$OUTPUT_DIR/" 2>/dev/null || true
    # Remove the now empty target directory
    rmdir "$OUTPUT_DIR/$TARGET" 2>/dev/null || true
fi

# # Normalize reconstruction to [0, 1] via peak normalization
# if [ -f "$OUTPUT_DIR/${TARGET}_r2d2.fits" ]; then
#     "$BASE_DIR/.envs/.benchmark/bin/python" -c "
# import sys; sys.path.append('$BASE_DIR/src')
# from utils.transforms import normalize_fits_peak
# normalize_fits_peak('$OUTPUT_DIR/${TARGET}_r2d2.fits')
# "
# fi

# Calculate beam parameters from the PSF, then format all output FITS
PSF_FILE="$OUTPUT_DIR/${TARGET}_r2d2_psf.fits"
if [ -f "$PSF_FILE" ]; then
    echo "Calculating beam parameters from PSF..."
    "$BASE_DIR/.envs/.benchmark/bin/python" -c "
import sys, glob; sys.path.append('$BASE_DIR/src')
from utils.transforms import calculate_beam, formatfits

# Fit the PSF to get beam parameters
beam = calculate_beam('$PSF_FILE', '$MS_FILE')
print(f\"Beam: bmaj={beam['bmaj']:.4f} arcsec, bmin={beam['bmin']:.4f} arcsec, bpa={beam['bpa']:.2f} deg\")

# Format all FITS files in the output directory with WCS + beam headers
for f in glob.glob('$OUTPUT_DIR/*.fits'):
    print(f'Formatting {f}...')
    formatfits(f, '$MS_FILE', beam_params=beam)
print('Done formatting all FITS files.')
" 2>&1 | tee -a "$LOG_FILE"
fi