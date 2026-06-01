#!/bin/bash

TARGET=disk_0004

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

PIXEL_SIZE=$($BASE_DIR/.envs/.benchmark/bin/python -c "import sys; sys.path.append('$BASE_DIR/src'); from utils.transforms import calculate_pixel_size; print(f'{calculate_pixel_size(\"$DATA_FILE\")}asec')")
echo $PIXEL_SIZE
start_time=$(date +%s.%N)

# Dynamically calculate the pixel scale based on the MS baselines
wsclean -size 64 64 \
    -scale $PIXEL_SIZE \
    -padding 1.2 \
    -weight briggs 2 \
    -multiscale \
    -mgain 0.8 \
    -gain 0.1 \
    -auto-threshold 1 \
    -auto-mask 3 \
    -nmiter 10 \
    -niter 100000 \
    -data-column DATA \
    -name $TARGET \
    "$DATA_FILE" 2>&1 | tee "$LOG_FILE"

mv "${TARGET}-image.fits" "${TARGET}_wsclean.fits"
$($BASE_DIR/.envs/.benchmark/bin/python -c "import sys; sys.path.append('$BASE_DIR/src'); from utils.transforms import convert_jybeam_to_jypixel; convert_jybeam_to_jypixel(\"$OUTPUT_DIR/${TARGET}_wsclean.fits\")")

end_time=$(date +%s.%N)
runtime=$(echo "$end_time - $start_time" | bc)

echo "Execution time: $runtime seconds" >> "$LOG_FILE"