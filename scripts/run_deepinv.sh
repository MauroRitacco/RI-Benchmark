#!/bin/bash
TARGET=disk_1111
N=1

BASE_DIR=$(pwd)
VENV_PYTHON="$BASE_DIR/.envs/.deepinv/bin/python"

if [ -d "$BASE_DIR/data/simulated/$TARGET" ]; then
    MS_FILE="$BASE_DIR/data/simulated/$TARGET/${TARGET}.MS"
    OUTPUT_DIR="$BASE_DIR/experiments/simulated/$TARGET/${TARGET}_deepinverse"
else
    MS_FILE="$BASE_DIR/data/archival/$TARGET/${TARGET}.MS"
    OUTPUT_DIR="$BASE_DIR/experiments/archival/$TARGET/${TARGET}_deepinverse"
fi

LOG_FILE="$OUTPUT_DIR/${TARGET}_deepinverse_script.log"

echo "Starting DeepInv for $TARGET..."

# Generate Briggs weights
./scripts/generate_weights.sh "$TARGET" "$N"

"$VENV_PYTHON" src/algorithms/deepinverse.py --target "$TARGET" | tee "$LOG_FILE"

# Calculate beam parameters from the PSF, then format all output FITS
PSF_FILE="$OUTPUT_DIR/${TARGET}_deepinverse_psf.fits"
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
