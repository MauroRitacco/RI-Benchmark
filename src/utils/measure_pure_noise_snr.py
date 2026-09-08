#!/usr/bin/env python3
"""Pure-noise dirty-image SNR per FOV class.

For each selected target:
  1. Copy its MS and replace the DATA column with complex unit-variance
     Gaussian noise (identical distribution to the dataset's additive noise).
  2. Run tclean niter=0 on the noise MS and on the real MS, with exactly the
     same imaging parameters as casa.py's dirty pass (briggs robust=2,
     standard gridder, 64x64, nominal pixel size).
  3. sigma_noise: corner-patch estimator (dirty_noise.estimate_noise) on the
     pure-noise dirty image; peak: max of the real dirty image;
     SNR = peak / sigma_noise.

Only the output CSV persists; every per-target working directory (copied MS,
dirty images, CASA logs) is deleted once the target is processed.
"""
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

OUT_DIR = BASE_DIR / "results" / "noise_snr"
CSV_PATH = OUT_DIR / "snr_pure_noise.csv"
CASA_PYTHON = BASE_DIR / ".envs" / ".casa" / "bin" / "python"

# FOV class -> number of targets to sample
CLASSES = {"s0p5": 20, "s2": 20, "s4": 20}

# Runs inside the CASA venv: writes pure noise into the copied MS and performs
# both dirty passes (noise MS and real MS) with casa.py's dirty-pass params.
CASA_WORKER = r"""
import os
import sys

import numpy as np
from casatasks import exportfits, tclean
from casatools import table

work, real_ms, noise_ms, pixel_size, seed = sys.argv[1:6]


def dirty_pass(ms, name):
    tclean(
        vis=ms,
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
        cyclefactor=0.0,
        savemodel="none",
    )
    # Same export logic as casa.py: keep only the dirty image as FITS.
    if os.path.exists(f"{name}.image"):
        os.rename(f"{name}.image", f"{name}.dirty")
    exportfits(
        imagename=f"{name}.dirty",
        fitsimage=f"{name}_dirty.fits",
        overwrite=True,
    )


# 1) Replace DATA with pure unit-variance complex Gaussian noise.
tb = table()
tb.open(noise_ms, nomodify=False)
data = tb.getcol("DATA")
rng = np.random.default_rng(int(seed))
noise = (
    rng.standard_normal(data.shape) + 1j * rng.standard_normal(data.shape)
) / np.sqrt(2.0)
tb.putcol("DATA", noise)
tb.close()

# 2) Dirty passes (noise MS and real MS).
dirty_pass(noise_ms, os.path.join(work, "noise"))
dirty_pass(real_ms, os.path.join(work, "real"))
"""

FIELDS = [
    "class",
    "target",
    "M",
    "sigma_theory_jy_per_beam",
    "sigma_measured_jy_per_beam",
    "dirty_peak_jy_per_beam",
    "snr_peak",
    "elapsed_s",
]


def main():
    import numpy as np
    from astropy.io import fits
    from scipy.io import loadmat

    from src.utils.dirty_noise import estimate_noise

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(CSV_PATH, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=FIELDS)
        writer.writeheader()
        for cls, n in CLASSES.items():
            targets = sorted(
                (BASE_DIR / "data" / "simulated").glob(f"disk_{cls}_*")
            )[:n]
            for tdir in targets:
                target = tdir.name
                work = OUT_DIR / f".work_{target}"
                t0 = time.perf_counter()
                row = None
                try:
                    ms = tdir / f"{target}.MS"
                    mat = loadmat(str(tdir / f"{target}.mat"))
                    pixel_size = float(mat["nominal_pixelsize"].item())
                    M = int(mat["y"].size)
                    seed = abs(hash(target)) % (2**32)

                    work.mkdir(parents=True)
                    noise_ms = work / f"{target}_noise.MS"
                    shutil.copytree(ms, noise_ms)

                    env = dict(os.environ, OMP_NUM_THREADS="1")
                    subprocess.run(
                        [
                            str(CASA_PYTHON), "-c", CASA_WORKER,
                            str(work), str(ms), str(noise_ms),
                            repr(pixel_size), str(seed),
                        ],
                        cwd=work,
                        env=env,
                        check=True,
                        capture_output=True,
                        text=True,
                    )

                    result = estimate_noise(
                        dirty_path=work / "noise_dirty.fits",
                        gt_path=None,
                        multiplier=1.0,
                    )
                    peak = float(
                        fits.getdata(str(work / "real_dirty.fits"))
                        .squeeze()
                        .max()
                    )
                    sigma = float(result.sigma_noise)
                    row = {
                        "class": cls,
                        "target": target,
                        "M": M,
                        "sigma_theory_jy_per_beam": f"{1.0 / np.sqrt(M):.6e}",
                        "sigma_measured_jy_per_beam": f"{sigma:.6e}",
                        "dirty_peak_jy_per_beam": f"{peak:.4f}",
                        "snr_peak": f"{peak / sigma:.1f}",
                        "elapsed_s": f"{time.perf_counter() - t0:.1f}",
                    }
                    print(
                        f"[{cls}] {target}: sigma={sigma:.4e} Jy/beam, "
                        f"peak={peak:.2f} Jy/beam, SNR={peak / sigma:.0f}"
                    )
                except subprocess.CalledProcessError as e:
                    print(f"[error] {target}: CASA worker failed:\n{e.stderr[-2000:]}")
                finally:
                    shutil.rmtree(work, ignore_errors=True)
                if row:
                    rows.append(row)
                    writer.writerow(row)
                    fcsv.flush()

    # Summary per class.
    print("\n=== Summary (mean +/- std over targets) ===")
    for cls in CLASSES:
        sub = [r for r in rows if r["class"] == cls]
        if not sub:
            continue
        sig = [float(r["sigma_measured_jy_per_beam"]) for r in sub]
        snr = [float(r["snr_peak"]) for r in sub]
        print(
            f"{cls:<6} n={len(sub):<3} sigma={np.mean(sig):.4e}+-{np.std(sig):.1e} "
            f"Jy/beam   SNR pico={np.mean(snr):.3g}+-{np.std(snr):.3g}"
        )
    print(f"\nCSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
