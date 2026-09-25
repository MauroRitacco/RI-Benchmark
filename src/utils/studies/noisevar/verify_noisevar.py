#!/usr/bin/env python3
"""Noisevar steps 2/3 verification: tclean dirty pass + checks.

  --mode noise  (linearity): for each noisevar/noisevar MS, tclean niter=0
      (briggs robust=2, standard gridder, 64x64, nominal pixel — the exact
      params used to build pure_noise_val360.csv) on the pure-noise MS and
      measure sigma_whole = nanstd of the dirty image (identical to the
      estimator behind the sigma_whole_jy_per_beam CSV column). Check:

          sigma_whole_medido / sigma_vis  ~=  sigma_whole_CSV   (+-10%)

      for the 3 levels of each target (linearity in sigma_vis).

  --mode signal (signal intact): for each noisevar/noisevar/snr<L> MS
      (original GT + noise at sigma_vis of level L), for the 3 levels, check:

          pico ~= dirty_peak_CSV                          (+-10%)
          pico / (sigma_whole_CSV * sigma_vis) ~= L       (+-5%)

Outputs results/noisevar/linearity_check.csv / signal_check.csv and prints
the tables. Runs inside the CASA venv:
  .envs/.casa/bin/python src/utils/studies/noisevar/verify_noisevar.py --mode noise
  .envs/.casa/bin/python src/utils/studies/noisevar/verify_noisevar.py --mode signal
"""
import argparse
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from casatasks import exportfits, tclean
from scipy.io import loadmat

BASE_DIR = Path(__file__).resolve().parents[4]
NOISEVAR = BASE_DIR / "data" / "noisevar"
OUT_DIR = BASE_DIR / "results" / "noisevar"
CSV_PATH = BASE_DIR / "results" / "noise_validation" / "bright" / "pure_noise_val360.csv"
CAL_CSV = OUT_DIR / "calibration_table.csv"

TOL = 0.10

NOISE_TARGETS = [
    "disk_s0p5_r1_c0_x0_000", "disk_s0p5_r1_c0_x0_001",
    "disk_s2_r1_c0_x0_000", "disk_s2_r1_c0_x0_001",
    "disk_s4_r1_c0_x0_000", "disk_s4_r1_c0_x0_001",
]
SIGNAL_TARGETS = [
    "disk_s0p5_r1_c0_x0_000",
    "disk_s2_r1_c0_x0_000",
    "disk_s4_r1_c0_x0_000",
]


def read_csv_index(path):
    index = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            index[row["target"]] = {
                "cls": row["class"],
                "sigma_whole": float(row["sigma_whole_jy_per_beam"]),
                "dirty_peak": float(row["dirty_peak_jy_per_beam"]),
            }
    return index


def read_calibration(path):
    cal = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            cal[(row["target"], int(row["nivel"]))] = float(row["sigma_vis"])
    return cal


def dirty_pass(ms, pixel_size, workdir):
    """tclean niter=0 dirty pass; returns path to the exported dirty FITS."""
    name = str(workdir / "dirty")
    tclean(
        vis=str(ms),
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
    if (workdir / "dirty.image").exists():
        (workdir / "dirty.image").rename(workdir / "dirty.dirty")
    exportfits(
        imagename=str(workdir / "dirty.dirty"),
        fitsimage=str(workdir / "dirty_dirty.fits"),
        overwrite=True,
    )
    return workdir / "dirty_dirty.fits"


def check_noise(targets):
    index = read_csv_index(CSV_PATH)
    cal = read_calibration(CAL_CSV)
    rows = []
    for target in targets:
        sigma_whole_csv = index[target]["sigma_whole"]
        cls = index[target]["cls"]
        for L in (50, 200, 500):
            name = f"{target}_snr{L}"
            tdir = NOISEVAR / "noisevar" / f"snr{L}" / name
            ms = tdir / f"{name}.MS"
            sigma_vis = cal[(target, L)]
            pixel = float(loadmat(str(tdir / f"{name}.mat"))["nominal_pixelsize"].item())
            with tempfile.TemporaryDirectory(dir=str(OUT_DIR)) as tmp:
                dirty_fits = dirty_pass(ms, pixel, Path(tmp))
                sigma_medido = float(np.nanstd(fits.getdata(str(dirty_fits)).squeeze()))
            esperado = sigma_whole_csv * sigma_vis
            ratio = sigma_medido / sigma_vis
            ok = abs(ratio / sigma_whole_csv - 1.0) <= TOL
            rows.append((cls, target, L, sigma_vis, esperado, sigma_medido, ratio, ok))
            print(f"[{cls}] {name}: sigma_vis={sigma_vis:.4e}  "
                  f"esperado={esperado:.4e}  medido={sigma_medido:.4e}  "
                  f"ratio={ratio:.4e}  {'OK' if ok else 'FAIL'}", flush=True)

    out_csv = OUT_DIR / "linearity_check.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "clase", "nivel", "sigma_vis",
                    "sigma_whole_esperado", "sigma_whole_medido", "ratio_medido_sobre_sigma_vis", "ok"])
        for cls, target, L, sigma_vis, esperado, medido, ratio, ok in rows:
            w.writerow([target, cls, L, f"{sigma_vis:.6e}", f"{esperado:.6e}",
                        f"{medido:.6e}", f"{ratio:.6e}", int(ok)])
    print(f"\nWrote {out_csv}")
    return rows, out_csv


def check_signal(targets):
    index = read_csv_index(CSV_PATH)
    cal = read_calibration(CAL_CSV)
    TOL_SNR = 0.05  # spot-check gate B: pico/sigma vs SNR objetivo
    rows = []
    for target in targets:
        cls = index[target]["cls"]
        sigma_whole_csv = index[target]["sigma_whole"]
        dirty_peak_csv = index[target]["dirty_peak"]
        for L in (50, 200, 500):
            name = f"{target}_snr{L}"
            tdir = NOISEVAR / "noisevar" / f"snr{L}" / name
            ms = tdir / f"{name}.MS"
            sigma_vis = cal[(target, L)]
            pixel = float(loadmat(str(tdir / f"{name}.mat"))["nominal_pixelsize"].item())
            with tempfile.TemporaryDirectory(dir=str(OUT_DIR)) as tmp:
                dirty_fits = dirty_pass(ms, pixel, Path(tmp))
                data = fits.getdata(str(dirty_fits)).squeeze()
                peak = float(np.nanmax(data))
            ratio_peak = peak / dirty_peak_csv
            snr_medido = peak / (sigma_whole_csv * sigma_vis)
            ok_peak = abs(ratio_peak - 1.0) <= TOL
            ok_snr = abs(snr_medido / L - 1.0) <= TOL_SNR
            ok = ok_peak and ok_snr
            rows.append((cls, target, L, sigma_vis, dirty_peak_csv, peak,
                         ratio_peak, snr_medido, ok))
            print(f"[{cls}] {name}: sigma_vis={sigma_vis:.4e}  "
                  f"peak_csv={dirty_peak_csv:.4e}  pico={peak:.4e}  "
                  f"pico/peak_csv={ratio_peak:.3f}  SNR={snr_medido:.1f} (obj {L})  "
                  f"{'OK' if ok else 'FAIL'}", flush=True)

    out_csv = OUT_DIR / "signal_check.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "clase", "nivel", "sigma_vis", "dirty_peak_csv",
                    "pico_medido", "pico_sobre_peak_csv", "snr_pico_medido", "ok"])
        for cls, target, L, sigma_vis, peak_csv, peak, ratio_peak, snr, ok in rows:
            w.writerow([target, cls, L, f"{sigma_vis:.6e}", f"{peak_csv:.6e}",
                        f"{peak:.6e}", f"{ratio_peak:.4f}", f"{snr:.2f}", int(ok)])
    print(f"\nWrote {out_csv}")
    return rows, out_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("noise", "signal"), required=True)
    parser.add_argument("--targets", nargs="+", default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "noise":
        rows, _ = check_noise(args.targets or NOISE_TARGETS)
        n_fail = sum(1 for r in rows if not r[7])
        print(f"\n{len(rows) - n_fail}/{len(rows)} passed "
              f"(sigma_medido/sigma_vis vs sigma_whole_CSV, tol +-10%)")
    else:
        rows, _ = check_signal(args.targets or SIGNAL_TARGETS)
        n_fail = sum(1 for r in rows if not r[8])
        print(f"\n{len(rows) - n_fail}/{len(rows)} passed "
              f"(pico vs dirty_peak +-10% y SNR vs objetivo +-5%)")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
