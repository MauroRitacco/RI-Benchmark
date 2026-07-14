#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <target>"
    exit 1
fi
TARGET=$1
N=${2:-2}

BASE_DIR=$(pwd)
WSCLEAN_VENV="$BASE_DIR/.envs/.wsclean/bin/activate"

if [ -d "$BASE_DIR/data/simulated/$TARGET" ]; then
    DATA_FILE="$BASE_DIR/data/simulated/$TARGET/${TARGET}.MS"
    OUTPUT_DIR="$BASE_DIR/experiments/simulated/$TARGET/${TARGET}_weights"
    LOG_FILE="$OUTPUT_DIR/${TARGET}_wsclean.log"
else
    DATA_FILE="$BASE_DIR/data/archival/$TARGET/${TARGET}.MS"
    OUTPUT_DIR="$BASE_DIR/experiments/archival/$TARGET/${TARGET}_weights"
    LOG_FILE="$OUTPUT_DIR/${TARGET}_wsclean.log"
fi

mkdir -p "$OUTPUT_DIR"

cd "$OUTPUT_DIR"

source $WSCLEAN_VENV
PIXEL_SIZE=$($BASE_DIR/.envs/.benchmark/bin/python -c "import sys; sys.path.append('$BASE_DIR/src'); from utils.transforms import calculate_pixel_size; print(f'{calculate_pixel_size(\"$DATA_FILE\", n=$N)}asec')")
echo $PIXEL_SIZE
echo "Starting WSClean for $TARGET..."
echo "Log saved in: $LOG_FILE"

start_time=$(date +%s.%N)

wsclean -size 64 64 \
    -scale $PIXEL_SIZE \
    -weight briggs 0 \
    -nmiter 0 \
    -niter 0 \
    -data-column DATA \
    -no-dirty \
    -name $TARGET \
    -store-imaging-weights \
    "$DATA_FILE" 2>&1 | tee "$LOG_FILE"


echo "Extracting IMAGING_WEIGHT_SPECTRUM and saving to ${TARGET}.mat..."
python3 -c "
import scipy.io as sio
from casatools import table

ms = '${DATA_FILE}'
mat = '${DATA_FILE%.MS}.mat'

tb = table()
tb.open(ms)
weights = tb.getcol('IMAGING_WEIGHT_SPECTRUM')
mask = tb.getcol('FLAG')[0, 0, :] == False
tb.close()

# Extract weights for unflagged data
w = weights[0, 0, mask]

# Append to .mat file
mat_data = sio.loadmat(mat)
mat_data['weight'] = w.reshape(-1, 1)
sio.savemat(mat, mat_data)
"
end_time=$(date +%s.%N)
runtime=$(echo "$end_time - $start_time" | bc)

echo -e "\nExecution time: $runtime seconds" >> "$LOG_FILE"

echo "✅ Proceso terminado. Tiempo total: $runtime seg."