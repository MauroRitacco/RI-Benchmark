#!/usr/bin/env python3
"""Noisevar steps 2/3: generate test MeasurementSets (GT fixed, noise varied).

Two modes:

  --mode noise  (linearity test): pure-noise MS, y = sigma_vis * CN(0,1),
      same base noise realization for all 3 levels of a target (per-target
      seed, same noise_seed() convention as the noises study). Output under
      data/noisevar/linearity/<target>_snr<L>/.

  --mode signal (signal-intact test): y = y_unit + sigma_vis(50) * CN(0,1),
      with y_unit the clean visibilities of the ORIGINAL unscaled GT through
      the same measurement operator/grid. Output under
      data/noisevar/signal/<target>_snr50/.

sigma_vis per (target, level) comes from results/noisevar/calibration_table.csv
(step 1). The MS is a copy of the original (data/simulated/<target>/, never
modified) with only the DATA column replaced on unflagged rows: same uv grid
and flags. Noise is generated in float64.

Each output dir gets <name>.MS, <name>.mat (y, nW, u, v, w, frequency,
nominal_pixelsize, weight — same convention as the noises dataset) and
<name>.json (mode, sigma_vis, snr_target).

Both modes write the MS with INDEPENDENT complex Gaussian noise per
correlation (each corr E|n|^2 = sigma_vis^2), matching exactly how
results/noise_validation/pure_noise_val360.csv was built. This matters:
the original MS duplicates the same visibility into both correlations, so
duplicating the noise would make tclean's Stokes-I average cancel the
sqrt(2) pol-combination gain and inflate the dirty image noise by sqrt(2)
relative to the CSV calibration. The .mat keeps the corr-0 realization
(single complex noise per visibility, the same visibility-space noise the
CSV convention defines); the MS corr-1 gets its own independent
realization.

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
    """Independent unit-variance complex noise per correlation (float64).

    Two draws from one per-target torch generator: corr-0 goes to the .mat
    `y` (and MS corr-0), corr-1 gets its own realization for the MS.
    """
    gen = torch.Generator().manual_seed(noise_seed(target))

    def draw():
        return (
            torch.randn(M, generator=gen, dtype=torch.float64)
            + 1j * torch.randn(M, generator=gen, dtype=torch.float64)
        ) / np.sqrt(2.0)

    return draw(), draw()


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


def write_cell(target, L, y_mat, y_ms_corr1, sigma_vis, mode, subdir):
    """Write <target>_snr<L>/ with MS (DATA replaced), .mat and .json.

    The .mat `y` (and MS corr-0) carries y_mat; MS corr-1 carries its own
    independent noise realization y_ms_corr1 (CSV convention).
    """
    name = f"{target}_snr{L}"
    outdir = OUT_ROOT / subdir / name
    if outdir.exists():
        shutil.rmtree(str(outdir))
    outdir.mkdir(parents=True)

    src = SIM_DATA / target
    mat = loadmat(str(src / f"{target}.mat"))
    savemat(str(outdir / f"{name}.mat"), {
        "y": np.asarray(y_mat).reshape(-1, 1),
        "nW": mat["nW"],
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
        "noise_convention": (
            "independent CN(0,sigma_vis^2) noise per MS correlation, float64, "
            "seed=noise_seed(target); .mat y = corr-0 realization"
        ),
    }
    if mode == "signal":
        meta["signal"] = "original unscaled GT (y_unit through MeasOpTkbNUFFT)"
    (outdir / f"{name}.json").write_text(json.dumps(meta, indent=2))

    # MS: copy of the original (same uvw/FLAG) with DATA replaced on
    # unflagged rows; corr-0 and corr-1 get independent noise realizations,
    # matching the pure-noise CSV measurement convention.
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
    ms_data[1, 0, mask] = np.asarray(y_ms_corr1).reshape(-1)
    tb.putcol("DATA", ms_data)
    tb.close()
    print(f"[{mode}] {name}: sigma_vis={sigma_vis:.6e} -> {outdir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("noise", "signal"), required=True)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--levels", nargs="+", type=int, default=None)
    args = parser.parse_args()

    cal = read_calibration(CAL_CSV)
    if args.mode == "noise":
        targets = args.targets or NOISE_TARGETS
        levels = tuple(args.levels or LEVELS)
        subdir = "linearity"
    else:
        targets = args.targets or SIGNAL_TARGETS
        levels = tuple(args.levels or (50,))
        subdir = "signal"

    for target in targets:
        if args.mode == "signal":
            _, y_unit = clean_visibilities(target, SIM_DATA / target / f"{target}.mat")
        n0, n1 = base_noises(target, loadmat(str(SIM_DATA / target / f"{target}.mat"))["u"].size)
        for L in levels:
            sigma_vis = cal[(target, L)]["sigma_vis"]
            if args.mode == "noise":
                y_mat = sigma_vis * n0.numpy()
                y_ms = sigma_vis * n1.numpy()
            else:
                y_mat = y_unit + sigma_vis * n0.numpy()
                y_ms = y_unit + sigma_vis * n1.numpy()
            write_cell(target, L, y_mat, y_ms, sigma_vis, args.mode, subdir)


if __name__ == "__main__":
    main()
