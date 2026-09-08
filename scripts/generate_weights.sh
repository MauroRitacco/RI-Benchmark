#!/bin/bash
# Generate Briggs imaging weights for DeepInverse and store them in <target>.mat.
# Temporary WSClean outputs are removed at the end.

if [ -z "$1" ]; then
    echo "Usage: $0 <target> [nval] [robust]"
    exit 1
fi
TARGET=$1
N=${2:-2}
ROBUST=${3:-2}

BASE_DIR=$(pwd)
WSCLEAN_VENV="$BASE_DIR/.envs/.wsclean/bin/activate"

if [ "${RI_BENCH_TEST:-0}" = "1" ]; then
    DATA_DIR="$BASE_DIR/data/simulated/test/$TARGET"
elif [ -d "$BASE_DIR/data/simulated/$TARGET" ]; then
    DATA_DIR="$BASE_DIR/data/simulated/$TARGET"
else
    DATA_DIR="$BASE_DIR/data/archival/$TARGET"
fi

DATA_FILE="$DATA_DIR/${TARGET}.MS"
MAT_FILE="$DATA_DIR/${TARGET}.mat"

# Temp WSClean working dir
TMP_DIR=$(mktemp -d "/tmp/${TARGET}_weights_XXXXXX")
LOG_FILE="$TMP_DIR/${TARGET}_wsclean.log"

echo "Generating DeepInverse weights for $TARGET ..."
echo "  MS: $DATA_FILE"
echo "  MAT: $MAT_FILE"
echo "  Temp dir: $TMP_DIR"

source $WSCLEAN_VENV
PIXEL_SIZE=$($BASE_DIR/.envs/.benchmark/bin/python -c "import sys; sys.path.append('$BASE_DIR/src'); from utils.transforms import calculate_pixel_size; print(f'{calculate_pixel_size(\"$DATA_FILE\", n=$N)}asec')")
echo "  Pixel size: $PIXEL_SIZE"

start_time=$(date +%s.%N)

cd "$TMP_DIR"

wsclean -size 64 64 \
    -scale $PIXEL_SIZE \
    -weight briggs $ROBUST \
    -nmiter 0 \
    -niter 0 \
    -data-column DATA \
    -no-dirty \
    -name $TARGET \
    -store-imaging-weights \
    "$DATA_FILE" > "$LOG_FILE" 2>&1

WSCLEAN_EXIT=$?

if [ $WSCLEAN_EXIT -ne 0 ]; then
    echo "ERROR: WSClean failed for $TARGET. Log left at $LOG_FILE"
    exit 1
fi

echo "Extracting IMAGING_WEIGHT_SPECTRUM and saving to $MAT_FILE ..."
$BASE_DIR/.envs/.wsclean/bin/python -c "
import scipy.io as sio
from casatools import table

ms = '${DATA_FILE}'
mat = '${MAT_FILE}'

tb = table()
tb.open(ms)
weights = tb.getcol('IMAGING_WEIGHT_SPECTRUM')
mask = tb.getcol('FLAG')[0, 0, :] == False
tb.close()

w = weights[0, 0, mask]

# Append to .mat file (overwriting only the 'weight' variable)
mat_data = sio.loadmat(mat)
mat_data['weight'] = w.reshape(-1, 1)
sio.savemat(mat, mat_data)
print(f'Saved weights for {len(w)} visibilities to {mat}')
"

PYTHON_EXIT=$?

end_time=$(date +%s.%N)
runtime=$(echo "$end_time - $start_time" | bc)

# Remove temp outputs; weights are already stored in the .mat
rm -rf "$TMP_DIR"

if [ $PYTHON_EXIT -ne 0 ]; then
    echo "ERROR: Failed to write weights to $MAT_FILE"
    exit 1
fi

echo "✅ Weights generated for $TARGET in ${runtime}s. Temporary files removed."
