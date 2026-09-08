#!/usr/bin/env python3
"""Validate the corner-patch noise estimator against pure-noise visibilities.

Builds a noise-only .mat/MS (y = (randn + 1j*randn)/sqrt(2) * tau, with
tau = 1/nW and M = #unflagged visibilities), generates tclean niter=0 dirty
images for both the noise-only and original MS, and compares the corner-patch
sigma_noise estimates against the theoretical tau/sqrt(M).

Run with the CASA Python environment:
    cd RI-Benchmark
    .envs/.casa/bin/python src/utils/validate_noise_estimator.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.io import loadmat, savemat

# Project imports
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from src.utils.dirty_noise import estimate_noise, plot_noise_validation

# CASA tools (only available inside the CASA Python environment)
import casatasks
from casatools import table


TARGET = "disk_s0p5_r1_c0_x0_005"
RANDOM_SEED = 42


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def copy_ms_replace_data(
    src_ms: Path,
    dst_ms: Path,
    new_y: np.ndarray,
) -> None:
    """Copy an MS and replace its DATA column with new_y on unflagged rows."""
    if dst_ms.exists():
        shutil.rmtree(dst_ms)
    shutil.copytree(src_ms, dst_ms)

    tb = table()
    tb.open(str(dst_ms), nomodify=False)
    try:
        data = tb.getcol("DATA")
        flag = tb.getcol("FLAG")
        mask = flag[0, 0, :] == False
        n_unflagged = int(mask.sum())
        if n_unflagged != new_y.size:
            raise ValueError(
                f"MS has {n_unflagged} unflagged rows but new_y has {new_y.size}"
            )
        data[:, 0, mask] = new_y.flatten()
        tb.putcol("DATA", data)
    finally:
        tb.close()


def generate_casa_dirty(
    vis_path: Path,
    out_name: Path,
    pixel_size: float,
) -> Path:
    """Run tclean niter=0 and export the dirty image to FITS."""
    if out_name.exists():
        shutil.rmtree(out_name)
    out_name.mkdir(parents=True, exist_ok=True)

    imagename = out_name / f"{out_name.name}_dirty"
    casatasks.casalog.setlogfile(str(out_name / f"{out_name.name}_dirty.log"))

    casatasks.tclean(
        vis=str(vis_path),
        imagename=str(imagename),
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

    dirty_fits = Path(f"{imagename}_dirty.fits")
    casatasks.exportfits(
        imagename=f"{imagename}.image",
        fitsimage=str(dirty_fits),
        overwrite=True,
    )
    return dirty_fits


def measure_sigma(
    dirty_fits: Path,
    gt_path: Path | None,
    multiplier: float = 1.0,
    detrend: bool = True,
) -> tuple[float, object]:
    """Return sigma_noise and the full NoiseResult."""
    result = estimate_noise(
        dirty_path=dirty_fits,
        gt_path=gt_path,
        multiplier=multiplier,
        detrend=detrend,
    )
    return result.sigma_noise, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "results" / "noise_validation")
    args = parser.parse_args()

    target = args.target
    base_dir = args.base_dir
    out_dir = ensure_dir(args.out_dir / f"{target}_{time.strftime('%Y%m%d_%H%M%S')}")

    orig_mat_path = base_dir / "data" / "simulated" / target / f"{target}.mat"
    orig_ms_path = base_dir / "data" / "simulated" / target / f"{target}.MS"
    gt_path = base_dir / "data" / "simulated" / target / f"{target}.fits"

    print(f"Target: {target}")
    print(f"Original .mat: {orig_mat_path}")
    print(f"Original MS:   {orig_ms_path}")
    print(f"Output dir:    {out_dir}")

    # 1. Load original .mat
    mat = loadmat(str(orig_mat_path))
    y_orig = mat["y"].flatten()
    nW = mat["nW"].flatten()
    u = mat["u"].flatten()
    v = mat["v"].flatten()
    w = mat["w"].flatten()
    frequency = float(mat["frequency"].item())
    pixel_size = float(mat["nominal_pixelsize"].item())
    weight = mat["weight"].flatten()

    tau = float(1.0 / nW[0])
    M = int(y_orig.size)
    sigma_theory = tau / np.sqrt(M)

    print(f"\n--- Visibility info ---")
    print(f"M (unflagged visibilities) = {M}")
    print(f"nW = {nW[0]:.6f}")
    print(f"tau = 1/nW = {tau:.6e}")
    print(f"Theoretical image noise (tau/sqrt(M)) = {sigma_theory:.6e} Jy/beam")

    # 2. Build pure-noise .mat
    rng = np.random.default_rng(RANDOM_SEED)
    y_noise = (rng.standard_normal(M) + 1j * rng.standard_normal(M)) / np.sqrt(2.0)
    y_noise = y_noise.astype(np.complex64) * tau

    noise_mat_path = out_dir / f"{target}_noise.mat"
    noise_mat = dict(mat)
    noise_mat["y"] = y_noise.reshape(-1, 1)
    savemat(str(noise_mat_path), noise_mat)
    print(f"\nSaved noise .mat: {noise_mat_path}")

    # 3. Create noise-only MS
    noise_ms_path = out_dir / f"{target}_noise.MS"
    print(f"Creating noise MS: {noise_ms_path}")
    copy_ms_replace_data(orig_ms_path, noise_ms_path, y_noise)

    # 4. Generate CASA dirty images
    print("\n--- Generating dirty images ---")
    noise_dirty_fits = generate_casa_dirty(
        noise_ms_path, out_dir / f"{target}_noise_casa", pixel_size
    )
    print(f"Noise dirty FITS: {noise_dirty_fits}")

    real_dirty_fits = generate_casa_dirty(
        orig_ms_path, out_dir / f"{target}_real_casa", pixel_size
    )
    print(f"Real dirty FITS: {real_dirty_fits}")

    # 5. Estimate sigma_noise
    print("\n--- Noise estimates ---")
    results = {}

    # Noise image, with detrending
    sigma_noise_det, res_noise_det = measure_sigma(
        noise_dirty_fits, gt_path=None, multiplier=1.0, detrend=True
    )
    results["noise_with_detrend"] = sigma_noise_det
    print(f"σ_noise (ruido puro, con detrend)  = {sigma_noise_det:.6e} Jy/beam")

    # Noise image, without detrending
    sigma_noise_not, res_noise_not = measure_sigma(
        noise_dirty_fits, gt_path=None, multiplier=1.0, detrend=False
    )
    results["noise_without_detrend"] = sigma_noise_not
    print(f"σ_noise (ruido puro, sin detrend) = {sigma_noise_not:.6e} Jy/beam")

    # Real image, with detrending and source filtering
    sigma_real, res_real = measure_sigma(
        real_dirty_fits, gt_path=gt_path, multiplier=1.0, detrend=True
    )
    results["real_with_detrend"] = sigma_real
    print(f"σ_real  (target real, con detrend)  = {sigma_real:.6e} Jy/beam")

    # 6. Diagnostics and summary
    print("\n--- Summary table ---")
    print(f"{'Case':<35} {'σ (Jy/beam)':>14} {'σ / σ_theory':>14}")
    print("-" * 65)
    for label, sigma in [
        ("noise pure (detrend)", sigma_noise_det),
        ("noise pure (no detrend)", sigma_noise_not),
        ("real target (detrend)", sigma_real),
    ]:
        print(f"{label:<35} {sigma:>14.6e} {sigma / sigma_theory:>14.2f}")
    print(f"{'theoretical tau/sqrt(M)':<35} {sigma_theory:>14.6e} {'1.00':>14}")

    # Save JSON summary
    summary = {
        "target": target,
        "M": M,
        "tau": tau,
        "sigma_theory": sigma_theory,
        "results": {
            "noise_with_detrend": sigma_noise_det,
            "noise_without_detrend": sigma_noise_not,
            "real_with_detrend": sigma_real,
        },
        "paths": {
            "noise_mat": str(noise_mat_path),
            "noise_ms": str(noise_ms_path),
            "noise_dirty_fits": str(noise_dirty_fits),
            "real_dirty_fits": str(real_dirty_fits),
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary JSON: {out_dir / 'summary.json'}")

    # Diagnostic PNGs
    try:
        plot_noise_validation(noise_dirty_fits, res_noise_det, out_dir / "noise_dirty_detrend.png")
        plot_noise_validation(noise_dirty_fits, res_noise_not, out_dir / "noise_dirty_no_detrend.png")
        plot_noise_validation(real_dirty_fits, res_real, out_dir / "real_dirty.png")
        print(f"Saved diagnostic PNGs in {out_dir}")
    except Exception as exc:
        print(f"Could not save diagnostic PNGs: {exc}")

    # Print decision criterion
    print("\n--- Decision criterion ---")
    if sigma_noise_det >= 0.5 * sigma_real:
        print("σ_noise ≈ σ_real  -> el estimador mide ruido térmico real.")
    else:
        print("σ_noise << σ_real -> el estimador está contaminado por sidelobes/residuo.")


if __name__ == "__main__":
    main()
