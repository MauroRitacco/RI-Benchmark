#!/bin/bash
TARGET=disk_1111
N=1

BASE_DIR=$(pwd)
VENV_PYTHON="$BASE_DIR/.envs/.casa/bin/python"

# tclean configuration dictionary (JSON format)
# Modify these parameters to configure the CASA tclean run
TCLEAN_CONFIG='{
    "imsize": 64,
    "specmode": "mfs",
    "deconvolver": "hogbom",
    "gridder": "standard",
    "weighting": "briggs",
    "robust": 2,
    "gain": 0.1,
    "niter": 1000000000,
    "threshold": "0.05Jy",
    "datacolumn": "data"
}'

echo "Starting casa for $TARGET..."

"$VENV_PYTHON" src/algorithms/casa.py --target "$TARGET" --config "$TCLEAN_CONFIG" --nval "$N"
