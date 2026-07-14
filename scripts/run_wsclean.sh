#!/bin/bash

TARGET=disk_1111
N=1

BASE_DIR=$(pwd)
WSCLEAN_VENV="$BASE_DIR/.envs/.wsclean/bin/activate"

if [ -d "$BASE_DIR/data/simulated/$TARGET" ]; then
    DATA_FILE="$BASE_DIR/data/simulated/$TARGET/${TARGET}.MS"
    OUTPUT_DIR="$BASE_DIR/experiments/simulated/$TARGET/${TARGET}_wsclean"
    LOG_FILE="$OUTPUT_DIR/${TARGET}_wsclean.log"
else
    DATA_FILE="$BASE_DIR/data/archival/$TARGET/${TARGET}.MS"
    OUTPUT_DIR="$BASE_DIR/experiments/archival/$TARGET/${TARGET}_wsclean"
    LOG_FILE="$OUTPUT_DIR/${TARGET}_wsclean.log"
fi

# Eliminate the directory to ensure a fresh run
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

cd "$OUTPUT_DIR"

source $WSCLEAN_VENV

echo "Starting WSClean for $TARGET..."
echo "Log saved in: $LOG_FILE"

# Old dynamic pixel-size calculation from the MS baselines
# PIXEL_SIZE=$($BASE_DIR/.envs/.benchmark/bin/python -c "import sys; sys.path.append('$BASE_DIR/src'); from utils.transforms import calculate_pixel_size; print(f'{calculate_pixel_size(\"$DATA_FILE\", n=$N)}asec')")
# echo $PIXEL_SIZE

# Load the nominal_pixelsize from the .mat file — this is the exact pixel scale used
# during visibility simulation, so WSClean images at the same angular scale as the groundtruth.
MAT_FILE="${DATA_FILE%.MS}.mat"
PIXEL_SIZE=$($BASE_DIR/.envs/.benchmark/bin/python -c "
from scipy.io import loadmat
pixel_size = float(loadmat('$MAT_FILE')['nominal_pixelsize'].item())
print(f'{pixel_size}asec')
")
echo "Pixel size (from .mat): $PIXEL_SIZE"
start_time=$(date +%s.%N)

# Dynamically calculate the pixel scale based on the MS baselines
WSCLEAN_CONFIG="-size 64 64 \
-scale $PIXEL_SIZE \
-padding 2 \
-j 1 \
-weight briggs 2 \
-multiscale \
-multiscale-scales 0,5,15 \
-mgain 0.8 \
-gain 0.1 \
-auto-threshold 0.5 \
-auto-mask 1.5 \
-niter 200000 \
-data-column DATA"

wsclean $WSCLEAN_CONFIG \
    -name $TARGET \
    "$DATA_FILE" 2>&1 | tee "$LOG_FILE"

mv "${TARGET}-image.fits" "${TARGET}_wsclean.fits"
$($BASE_DIR/.envs/.benchmark/bin/python -c "import sys; sys.path.append('$BASE_DIR/src'); from utils.transforms import convert_jybeam_to_jypixel; convert_jybeam_to_jypixel(\"$OUTPUT_DIR/${TARGET}_wsclean.fits\")")

# Normalize to [0, 1] via peak normalization
# $($BASE_DIR/.envs/.benchmark/bin/python -c "import sys; sys.path.append('$BASE_DIR/src'); from utils.transforms import normalize_fits_peak; normalize_fits_peak(\"$OUTPUT_DIR/${TARGET}_wsclean.fits\")")

end_time=$(date +%s.%N)
runtime=$(echo "$end_time - $start_time" | bc)

echo "wsclean config: $WSCLEAN_CONFIG" >> "$LOG_FILE"
echo "Execution time: $runtime seconds" >> "$LOG_FILE"