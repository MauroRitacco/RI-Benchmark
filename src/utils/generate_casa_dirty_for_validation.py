#!/usr/bin/env python3
"""Generate CASA dirty images (tclean niter=0) for validation targets.

The dirty images are written to a validation sub-directory and are NOT used by
production reconstructions.  They are only meant to test the noise-estimation
helper before it is integrated into casa.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Add project root for imports
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

import casatasks
from scipy.io import loadmat


def generate_dirty(target: str, base_dir: Path, out_dir: Path) -> Path:
    """Run tclean with niter=0 and export the dirty image to FITS."""
    category = "simulated" if (base_dir / "data" / "simulated" / target).exists() else "archival"
    vis_path = str(base_dir / "data" / category / target / f"{target}.MS")
    mat_path = vis_path.replace(".MS", ".mat")
    pixel_size = float(loadmat(mat_path)["nominal_pixelsize"].item())

    cell_dir = out_dir / target
    if cell_dir.exists():
        shutil.rmtree(cell_dir)
    cell_dir.mkdir(parents=True, exist_ok=True)

    name = str(cell_dir / f"{target}_casa_dirty")
    casatasks.casalog.setlogfile(str(cell_dir / f"{target}_casa_dirty.log"))

    casatasks.tclean(
        vis=vis_path,
        imagename=name,
        imsize=64,
        cell=f"{pixel_size}arcsec",
        specmode="mfs",
        deconvolver="hogbom",
        gridder="standard",
        weighting="briggs",
        robust=2,
        stokes="I",
        datacolumn="data",
        niter=0,
    )

    dirty_fits = Path(f"{name}_dirty.fits")
    casatasks.exportfits(
        imagename=f"{name}.image",
        fitsimage=str(dirty_fits),
        overwrite=True,
    )
    return dirty_fits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "results" / "dirty_noise_validation" / "casa_dirty")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for target in args.targets:
        print(f"Generating CASA dirty image for {target} ...")
        dirty_fits = generate_dirty(target, args.base_dir, args.out_dir)
        index[target] = str(dirty_fits)
        print(f"  -> {dirty_fits}")

    with open(args.out_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)


if __name__ == "__main__":
    main()
