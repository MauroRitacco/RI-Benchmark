#!/usr/bin/env python3
"""Noisevar steps 2/3: generate test MeasurementSets (GT fixed, noise varied).

Two modes:

  --mode noise  (linearity test): pure-noise MS, y = sigma_vis * CN(0,1),
      same base noise realization for all 3 levels of a target (per-target
      seed, same noise_seed() convention as the noises study). Output under
      data/noisevar/noisevar/<target>_snr<L>/.

  --mode signal (signal+noise): y = y_unit + sigma_vis * CN(0,1),
      with y_unit the clean visibilities of the ORIGINAL unscaled GT through
      the same measurement operator/grid. Gate-B cells (1 target/class,
      snr50 only) go to data/noisevar/signal/; the full dataset (all levels)
      with --snr-subdirs goes to data/noisevar/noisevar/snr<L>/ (in-place
      overwrite of the noise-mode cells). Each .json carries gt_fits: the
      absolute path to the original unscaled GT FITS (data/simulated/).

sigma_vis per (target, level) comes from results/noisevar/calibration_table.csv
(step 1). The MS is a copy of the original (data/simulated/<target>/, never
modified) with only the DATA column replaced on unflagged rows: same uv grid
and flags. Noise is generated in float64.

Each output dir gets <name>.MS, <name>.mat (y, nW, u, v, w, frequency,
nominal_pixelsize, weight — same convention as the noises dataset) and
<name>.json (mode, sigma_vis, snr_target).

CONVENCION FINAL DE RUIDO (2026-09-14): XX=YY identicos — una unica
realizacion de ruido gaussiano complejo por visibilidad (desvio sigma_vis),
escrita igual en ambas correlaciones del MS y en el .mat ("Stokes I
efectivo"). Consecuencia: nW = 1/sigma_vis en el .mat (no se copia de
datasets viejos). R2D2: la escala de nW se cancela en normalize_instance.
DeepInv: usa nW en su likelihood, necesita el valor correcto.

Fixed verification targets (first two of each class, deterministic):
    s0p5: disk_s0p5_r1_c0_x0_000, disk_s0p5_r1_c0_x0_001
    s2:   disk_s2_r1_c0_x0_000,   disk_s2_r1_c0_x0_001
    s4:   disk_s4_r1_c0_x0_000,   disk_s4_r1_c0_x0_001
signal mode uses the first of each class.

Runs inside the DiskGenerator venv (torch + torchkbnufft + casatools):
  DiskGenerator/.venv/bin/python src/utils/studies/noisevar/generate_noisevar_ms.py --mode noise
  DiskGenerator/.venv/bin/python src/utils/studies/noisevar/generate_noisevar_ms.py --mode signal
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from scipy.io import loadmat, savemat

BASE_DIR = Path(__file__).resolve().parents[4]          # RI-Benchmark
THESIS_DIR = BASE_DIR.parent                            # /home/mauro/projects/Thesis
DISKGEN = THESIS_DIR / "DiskGenerator"
SIM_DATA = BASE_DIR / "data" / "simulated"
OUT_ROOT = BASE_DIR / "data" / "noisevar"
CAL_CSV = BASE_DIR / "results" / "noisevar" / "calibration_table.csv"

sys.path.insert(0, str(DISKGEN / "utils" / "ri_measurement_operator"))
sys.path.insert(0, str(DISKGEN / "studies" / "faint_disks"))
from pysrc.utils.io import load_data_to_tensor  # noqa: E402
from pysrc.measOperator.meas_op_nufft_tkbn import MeasOpTkbNUFFT  # noqa: E402
from rescale_and_simulate import noise_seed  # noqa: E402

IMG_SIZE = (64, 64)
LEVELS = (50, 200, 500)

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


def read_calibration(path):
    """(target, nivel) -> sigma_vis, clase from the step-1 table."""
    cal = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            cal[(row["target"], int(row["nivel"]))] = {
                "cls": row["clase"],
                "sigma_vis": float(row["sigma_vis"]),
            }
    return cal


def base_noises(target, M):
    """Single unit-variance complex noise realization (float64), XX=YY.

    One draw from one per-target torch generator (same seed convention as
    the noises study); written identically to MS corr-0/corr-1 and the .mat.
    """
    gen = torch.Generator().manual_seed(noise_seed(target))
    n0 = (
        torch.randn(M, generator=gen, dtype=torch.float64)
        + 1j * torch.randn(M, generator=gen, dtype=torch.float64)
    ) / np.sqrt(2.0)
    return n0, n0


def clean_visibilities(target, mat_path):
    """y_unit: forward operator of the ORIGINAL unscaled GT (signal mode)."""
    mat = loadmat(str(mat_path))
    data = load_data_to_tensor(
        uv_file_path=str(mat_path),
        super_resolution=1,
        img_size=IMG_SIZE,
        image_pixel_size=float(mat["nominal_pixelsize"].item()),
        weight_robustness=2.0,
    )
    meas_op = MeasOpTkbNUFFT(u=data["u"], v=data["v"], img_size=IMG_SIZE)
    gt = fits.getdata(str(SIM_DATA / target / f"{target}.fits")).astype(float)
    gt_t = torch.tensor(gt, dtype=torch.float64).view(1, 1, *gt.shape)
    with torch.no_grad():
        y_unit = meas_op.forward_op(gt_t)
    if y_unit.numel() != int(mat["u"].size):
        raise RuntimeError(f"{target}: operator size {y_unit.numel()} != M {mat['u'].size}")
    return mat, y_unit.numpy()


def write_cell(target, L, y_mat, sigma_vis, mode, subdir, snr_subdirs=False):
    """Write <target>_snr<L>/ with MS (DATA replaced), .mat and .json.

    XX=YY convention: the .mat `y` and BOTH MS correlations carry the same
    realization y_mat ("effective Stokes I"); nW = 1/sigma_vis.
    """
    name = f"{target}_snr{L}"
    outdir = OUT_ROOT / subdir / (f"snr{L}" if snr_subdirs else "") / name
    if outdir.exists():
        shutil.rmtree(str(outdir))
    outdir.mkdir(parents=True)

    src = SIM_DATA / target
    mat = loadmat(str(src / f"{target}.mat"))
    savemat(str(outdir / f"{name}.mat"), {
        "y": np.asarray(y_mat).reshape(-1, 1),
        "nW": np.full((int(np.asarray(y_mat).size), 1), 1.0 / sigma_vis,
                      dtype=np.float32),
        "u": mat["u"],
        "v": mat["v"],
        "w": mat["w"],
        "frequency": float(mat["frequency"].item()),
        "nominal_pixelsize": float(mat["nominal_pixelsize"].item()),
        "weight": mat["weight"],
    })

    meta = {
        "name": name,
        "mode": mode,
        "snr_target": L,
        "sigma_vis": sigma_vis,
        "noise_seed_target": target,
        "gt_fits": str(SIM_DATA / target / f"{target}.fits"),
        "noise_convention": (
            "XX=YY single CN(0,sigma_vis^2) realization, float64, "
            "seed=noise_seed(target); .mat y = same realization; "
            "nW = 1/sigma_vis (effective Stokes I)"
        ),
    }
    if mode == "signal":
        meta["signal"] = "original unscaled GT (y_unit through MeasOpTkbNUFFT)"
    (outdir / f"{name}.json").write_text(json.dumps(meta, indent=2))

    # MS: copy of the original (same uvw/FLAG) with DATA replaced on
    # unflagged rows; corr-0 and corr-1 get the SAME noise realization
    # (XX=YY final convention).
    new_ms = outdir / f"{name}.MS"
    shutil.copytree(str(src / f"{target}.MS"), str(new_ms))
    from casatools import table
    tb = table()
    tb.open(str(new_ms), nomodify=False)
    ms_data = tb.getcol("DATA")
    mask = tb.getcol("FLAG")[0, 0, :] == False  # noqa: E712
    M = int(np.asarray(y_mat).reshape(-1).shape[0])
    if int(mask.sum()) != M:
        tb.close()
        raise RuntimeError(f"{name}: unflagged rows {int(mask.sum())} != M {M}")
    ms_data[0, 0, mask] = np.asarray(y_mat).reshape(-1)
    ms_data[1, 0, mask] = np.asarray(y_mat).reshape(-1)
    tb.putcol("DATA", ms_data)
    tb.close()
    print(f"[{mode}] {name}: sigma_vis={sigma_vis:.6e} -> {outdir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("noise", "signal"), required=True)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--levels", nargs="+", type=int, default=None)
    parser.add_argument("--snr-subdirs", action="store_true",
                        help="write under <subdir>/snr<L>/ (dataset layout)")
    parser.add_argument("--out-subdir", default=None,
                        help="override output subdir (default: noisevar/signal by mode)")
    args = parser.parse_args()

    cal = read_calibration(CAL_CSV)
    if args.mode == "noise":
        targets = args.targets or NOISE_TARGETS
        levels = tuple(args.levels or LEVELS)
        subdir = "noisevar"
    else:
        targets = args.targets or SIGNAL_TARGETS
        levels = tuple(args.levels or (50,))
        subdir = "signal"
    if args.out_subdir:
        subdir = args.out_subdir

    for target in targets:
        if args.mode == "signal":
            _, y_unit = clean_visibilities(target, SIM_DATA / target / f"{target}.mat")
        n0, _ = base_noises(target, loadmat(str(SIM_DATA / target / f"{target}.mat"))["u"].size)
        for L in levels:
            sigma_vis = cal[(target, L)]["sigma_vis"]
            if args.mode == "noise":
                y_mat = sigma_vis * n0.numpy()
            else:
                y_mat = y_unit + sigma_vis * n0.numpy()
            write_cell(target, L, y_mat, sigma_vis, args.mode, subdir,
                       snr_subdirs=args.snr_subdirs)


if __name__ == "__main__":
    main()
