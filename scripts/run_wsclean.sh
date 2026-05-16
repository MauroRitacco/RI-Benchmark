#!/bin/bash

TARGET=disk_0000

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

mkdir -p "$OUTPUT_DIR"

cd "$OUTPUT_DIR"

source $WSCLEAN_VENV

echo "Starting WSClean for $TARGET..."
echo "Log saved in: $LOG_FILE"

start_time=$(date +%s.%N)

wsclean -size 64 64 \
    -scale 8.44e-1asec \
    -weight briggs 0 \
    -mgain 1 -gain 0.1 \
    -threshold 0.001Jy \
    -auto-mask 1 \
    -padding 2 \
    -nmiter 1 \
    -niter 500000 \
    -data-column DATA \
    -name $TARGET \
    "$DATA_FILE" 2>&1 | tee "$LOG_FILE"

mv "${TARGET}-image.fits" "${TARGET}_wsclean.fits"

end_time=$(date +%s.%N)
runtime=$(echo "$end_time - $start_time" | bc)

echo -e "\nExecution time: $runtime seconds" >> "$LOG_FILE"

echo "✅ Proceso terminado. Tiempo total: $runtime seg."