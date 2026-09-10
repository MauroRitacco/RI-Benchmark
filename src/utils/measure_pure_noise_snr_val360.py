#!/usr/bin/env python3
"""Pure-noise dirty-image SNR over the full 360-target validation set.

Identical methodology to measure_pure_noise_snr.py (60-target sweep_val
study): copy each MS, replace DATA with complex unit-variance Gaussian noise
(deterministic seed per target name), tclean niter=0 dirty passes on the
noise MS and the real MS with casa.py's dirty-pass params (briggs robust=2,
standard gridder, 64x64, nominal pixel), corner-patch estimator with and
without plane detrending, whole-image std/rms, real-dirty peak and SNR.

Differences with the 60-target script: all 120 targets per FOV class from
data/simulated/ (never data/simulated/test/), 4 parallel workers, progress
log every 10 targets, and a per-class summary written to a .txt file.

Only the output CSV and summary persist; every per-target working directory
(copied MS, dirty images, CASA logs) is deleted once the target is processed.
"""
import argparse
import csv
import hashlib
import os
import multiprocessing as mp
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

OUT_DIR = BASE_DIR / "results" / "noise_validation"
CSV_PATH = OUT_DIR / "pure_noise_val360.csv"
SUMMARY_PATH = OUT_DIR / "pure_noise_val360_summary.txt"
CASA_PYTHON = BASE_DIR / ".envs" / ".casa" / "bin" / "python"

# FOV class -> number of validation targets
CLASSES = {"s0p5": 120, "s2": 120, "s4": 120}

WORKERS = 4

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
    "sigma_measured_jy_per_beam",
    "sigma_corners_nodetrend_jy_per_beam",
    "sigma_whole_jy_per_beam",
    "rms_corners_jy_per_beam",
    "rms_corners_nodetrend_jy_per_beam",
    "rms_whole_jy_per_beam",
    "sigma_real_jy_per_beam",
    "sigma_real_nodetrend_jy_per_beam",
    "rms_real_jy_per_beam",
    "rms_real_nodetrend_jy_per_beam",
    "dirty_peak_jy_per_beam",
    "snr_peak",
    "elapsed_s",
]


def process_target(task):
    """Process one target; returns (cls, target, row, error)."""
    import numpy as np
    from astropy.io import fits
    from scipy.io import loadmat

    from src.utils.dirty_noise import estimate_noise

    cls, tdir_str = task
    tdir = Path(tdir_str)
    target = tdir.name
    work = OUT_DIR / f".work_{target}"
    t0 = time.perf_counter()
    try:
        ms = tdir / f"{target}.MS"
        mat = loadmat(str(tdir / f"{target}.mat"))
        pixel_size = float(mat["nominal_pixelsize"].item())
        M = int(mat["y"].size)
        # Deterministic seed: Python's builtin hash() is randomized
        # per process, which made past noise realizations irreproducible.
        seed = int.from_bytes(
            hashlib.sha256(target.encode()).digest()[:4], "little"
        )

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

        noise_data = fits.getdata(str(work / "noise_dirty.fits")).squeeze()
        sigma_whole = float(np.nanstd(noise_data))
        result = estimate_noise(
            dirty_path=work / "noise_dirty.fits",
            gt_path=None,
            multiplier=1.0,
        )
        # Same corner estimator without plane detrending, to separate
        # the detrending bias from true noise non-stationarity.
        result_nd = estimate_noise(
            dirty_path=work / "noise_dirty.fits",
            gt_path=None,
            multiplier=1.0,
            detrend=False,
        )
        peak = float(
            fits.getdata(str(work / "real_dirty.fits"))
            .squeeze()
            .max()
        )
        sigma = float(result.sigma_noise)
        sigma_nd = float(result_nd.sigma_noise)
        rms = float(np.mean([p.rms for p in result.valid_patches]))
        rms_nd = float(np.mean([p.rms for p in result_nd.valid_patches]))
        rms_whole = float(np.sqrt(np.nanmean(noise_data**2)))
        # Same corner estimator on the REAL dirty image (y=Ax+n):
        # shows how much the corner patches inflate due to sidelobe
        # pedestal / residual source structure on the bright data.
        try:
            res_real = estimate_noise(
                dirty_path=work / "real_dirty.fits",
                gt_path=None,
                multiplier=1.0,
            )
            res_real_nd = estimate_noise(
                dirty_path=work / "real_dirty.fits",
                gt_path=None,
                multiplier=1.0,
                detrend=False,
            )
            sigma_real = float(res_real.sigma_noise)
            sigma_real_nd = float(res_real_nd.sigma_noise)
            rms_real = float(
                np.mean([p.rms for p in res_real.valid_patches])
            )
            rms_real_nd = float(
                np.mean([p.rms for p in res_real_nd.valid_patches])
            )
        except Exception as e:
            print(f"[warn] {target}: real-dirty corner stats failed: {e}")
            sigma_real = sigma_real_nd = rms_real = rms_real_nd = ""
        row = {
            "class": cls,
            "target": target,
            "M": M,
            "sigma_measured_jy_per_beam": f"{sigma:.6e}",
            "sigma_corners_nodetrend_jy_per_beam": f"{sigma_nd:.6e}",
            "sigma_whole_jy_per_beam": f"{sigma_whole:.6e}",
            "rms_corners_jy_per_beam": f"{rms:.6e}",
            "rms_corners_nodetrend_jy_per_beam": f"{rms_nd:.6e}",
            "rms_whole_jy_per_beam": f"{rms_whole:.6e}",
            "sigma_real_jy_per_beam": (
                f"{sigma_real:.6e}" if sigma_real != "" else ""
            ),
            "sigma_real_nodetrend_jy_per_beam": (
                f"{sigma_real_nd:.6e}" if sigma_real_nd != "" else ""
            ),
            "rms_real_jy_per_beam": (
                f"{rms_real:.6e}" if rms_real != "" else ""
            ),
            "rms_real_nodetrend_jy_per_beam": (
                f"{rms_real_nd:.6e}" if rms_real_nd != "" else ""
            ),
            "dirty_peak_jy_per_beam": f"{peak:.4f}",
            "snr_peak": f"{peak / sigma:.1f}",
            "elapsed_s": f"{time.perf_counter() - t0:.1f}",
        }
        real_str = (
            f"real: sigma={sigma_real:.4e}/{sigma_real_nd:.4e} "
            f"rms={rms_real:.4e}/{rms_real_nd:.4e}, "
        ) if sigma_real != "" else "real: sigma=-, rms=-, "
        print(
            f"[{cls}] {target}: sigma={sigma:.4e}/{sigma_nd:.4e} "
            f"(corners detrend/no), sigma_whole={sigma_whole:.4e}, "
            f"rms={rms:.4e}/{rms_nd:.4e}/{rms_whole:.4e}, "
            f"{real_str}"
            f"peak={peak:.2f} Jy/beam, SNR={peak / sigma:.0f}",
            flush=True,
        )
        return cls, target, row, None
    except subprocess.CalledProcessError as e:
        return cls, target, None, f"CASA worker failed:\n{e.stderr[-2000:]}"
    except Exception as e:
        return cls, target, None, f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def build_tasks(only_targets=None):
    tasks = []
    for cls, n in CLASSES.items():
        targets = sorted(
            (BASE_DIR / "data" / "simulated").glob(f"disk_{cls}_*")
        )[:n]
        for tdir in targets:
            if only_targets and tdir.name not in only_targets:
                continue
            tasks.append((cls, str(tdir)))
    return tasks


def mean_std(values):
    import numpy as np
    a = np.asarray(values, dtype=float)
    return float(np.mean(a)), float(np.std(a))


def write_summary(rows, summary_path, csv_path):
    import numpy as np

    lines = [
        "Pure-noise validation over 360 validation targets",
        f"CSV: {csv_path}",
        "sigmas in Jy/beam; ratios are per-target means.",
        "",
    ]
    for cls in CLASSES:
        sub = [r for r in rows if r["class"] == cls]
        if not sub:
            continue
        sig = [float(r["sigma_measured_jy_per_beam"]) for r in sub]
        signd = [float(r["sigma_corners_nodetrend_jy_per_beam"]) for r in sub]
        sigw = [float(r["sigma_whole_jy_per_beam"]) for r in sub]
        rmsc = [float(r["rms_corners_jy_per_beam"]) for r in sub]
        rmsnd = [float(r["rms_corners_nodetrend_jy_per_beam"]) for r in sub]
        rmsw = [float(r["rms_whole_jy_per_beam"]) for r in sub]
        m_sig, s_sig = mean_std(sig)
        ratio_dt_nd = [a / b for a, b in zip(sig, signd)]
        ratio_cw = [a / b for a, b in zip(signd, sigw)]
        ratio_rms_std_patch = [a / b for a, b in zip(rmsnd, signd)]
        ratio_rms_std_whole = [a / b for a, b in zip(rmsw, sigw)]

        def fmt(vals):
            return f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}"

        lines += [
            f"[{cls}] n={len(sub)}",
            f"  sigma_measured (detrend):    {m_sig:.4e} +/- {s_sig:.4e} Jy/beam",
            f"  sigma detrend/nodetrend:     {fmt(ratio_dt_nd)}",
            f"  corners/whole (nodetrend):   {fmt(ratio_cw)}",
            f"  rms/std patch (nodetrend):   {fmt(ratio_rms_std_patch)}",
            f"  rms/std whole image:         {fmt(ratio_rms_std_whole)}",
            "",
        ]
    text = "\n".join(lines)
    summary_path.write_text(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets", nargs="*", default=None,
        help="Run only these target names (verification runs).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Override the output CSV path (default: %(default)s).",
    )
    args = parser.parse_args()

    csv_path = Path(args.out) if args.out else CSV_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(only_targets=set(args.targets) if args.targets else None)
    print(f"Targets to process: {len(tasks)} with {WORKERS} workers")

    rows = []
    done = 0
    errors = []
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=FIELDS)
        writer.writeheader()
        with mp.Pool(WORKERS) as pool:
            for cls, target, row, err in pool.imap_unordered(
                process_target, tasks, chunksize=1
            ):
                done += 1
                if err:
                    errors.append((target, err))
                    print(f"[error] {target}: {err}")
                elif row:
                    rows.append(row)
                    writer.writerow(row)
                    fcsv.flush()
                if done % 10 == 0 or done == len(tasks):
                    print(
                        f"[progress] {done}/{len(tasks)} targets done, "
                        f"{len(errors)} errors",
                        flush=True,
                    )

    print(f"\nCSV: {csv_path}")
    if errors:
        print(f"{len(errors)} targets failed:")
        for target, err in errors:
            print(f"  {target}: {err[:200]}")

    summary_path = csv_path.with_name(
        csv_path.stem + "_summary.txt"
    )
    write_summary(rows, summary_path, csv_path)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
