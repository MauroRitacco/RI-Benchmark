#!/usr/bin/env python3
"""Orchestrator for archival (real, no ground truth) reconstructions.

One target at a time. Modes:

  --dirty   Generate only the full-resolution dirty image (cell from the
            longest baseline of the MS) with CASA tclean (niter=0), saved
            at target level in experiments/archival/<target>/ together with
            a parameter log. Kept in Jy/beam so its rms gives the absolute
            sigma threshold for casa/wsclean directly.

  --casa    CASA tclean reconstruction (hogbom, same grid as the dirty,
            3*sigma absolute threshold). The absolute threshold (3*sigma,
            Jy/beam) is passed directly via --threshold. Output (Jy/pixel)
            goes to experiments/archival/<target>/<target>_casa/ with a log.

  --wsclean WSClean reconstruction (single-scale clean, hogbom-equivalent,
            same grid + threshold convention as --casa). Output (Jy/pixel)
            goes to experiments/archival/<target>/<target>_wsclean/ with a
            log.

  --deepinv DeepInverse reconstruction (FISTA + WaveletPrior, archival-yaml
            config l1_i50_v2 + fluxcal) on the no-cutoff mat at the accepted
            r2d2 grid (64x64 @ ~0.0476 arcsec): the mat is copied with
            nominal_pixelsize rescaled (uv unchanged; declared). Weighting is
            natural only (mat nW), briggs=1 — declared deviation: wsclean 3.7
            cannot persist briggs imaging weights on this multi-spw MS, while
            casa/wsclean image with briggs robust=2. Output (Jy/pixel) goes to
            experiments/archival/<target>/<target>_deepinverse/ with a log.

  --png     Render only the comparison panel PNG (no reconstruction) from the
            existing products: DSHARP published reference + casa + wsclean +
            deepinv + r2d2. Faithful single-row port of the noisevar mold
            (src/utils/studies/noisevar/plot_noisevar_comparisons.py,
            render_all_preview, compact geometry, own colorbar per panel,
            log-count histogram under each panel). Saved to
            results/archival/<target>_panel.png. All images Jy/pixel (DSHARP
            converted from Jy/beam with its header beam). No GT, no error
            row, no metrics.

Run with the benchmark env (casacore + astropy + scipy):

    .envs/.benchmark/bin/python src/archival_recon.py --target <name> --dirty

Units convention (same as the rest of the pipeline): reconstruction FITS are
in Jy/pixel. The dirty image is kept in Jy/beam (native CASA units) because
the absolute sigma threshold for casa/wsclean is set from its rms: sigma
[Jy/beam] -> 3*sigma threshold, no conversion needed. Reconstructions are
converted with the beam-area formula used by src/utils/transforms.py
(convert_jybeam_to_jypixel).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_DIR = BASE_DIR / "data" / "archival"
EXPERIMENTS_DIR = BASE_DIR / "experiments" / "archival"
CASA_ENV = BASE_DIR / ".envs" / ".casa" / "bin" / "python"

# HD163296_continuum.MS has DATA + MODEL_DATA only (no CORRECTED_DATA);
# the calibrated data lives in DATA.
DEFAULT_DATA_COLUMN = "DATA"
DEFAULT_IMSIZE = 1024
DEFAULT_ROBUST = 2.0


def target_paths(target: str) -> tuple[Path, Path]:
    """Resolve and validate the MS and .mat of an archival target."""
    tdir = DATA_DIR / target
    ms = tdir / f"{target}.MS"
    mat = tdir / f"{target}.mat"
    if not ms.is_dir():
        raise SystemExit(f"MS not found: {ms}")
    if not mat.exists():
        raise SystemExit(f".mat not found: {mat} (src/utils/ms2mat.py)")
    return ms, mat


def cell_from_ms(ms_path: Path) -> tuple[float, float]:
    """Pixel size (arcsec, n=2) from the longest projected baseline of the MS.

    Returns (cell_arcsec, max_baseline_lambda).
    """
    from casacore import tables

    with tables.table(str(ms_path)) as t:
        uvw = t.getcol("UVW")
    with tables.table(str(ms_path) + "/SPECTRAL_WINDOW") as spw:
        freqs = np.array([spw.getcell("CHAN_FREQ", i)[0] for i in range(spw.nrows())])
    bl_lambda = np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2).max() * freqs.mean() / 299792458.0
    cell = (180.0 * 3600.0 / np.pi) / (2.0 * bl_lambda)
    return float(cell), float(bl_lambda)


def cell_from_mat(mat_path: Path) -> float:
    """nominal_pixelsize stored by src/utils/ms2mat.py (uv_cutoff dependent)."""
    from scipy.io import loadmat

    return float(loadmat(mat_path)["nominal_pixelsize"].item())


def resolve_cell(args, ms: Path, mat: Path) -> tuple[float, str]:
    if args.pixelsize is not None:
        return args.pixelsize, f"manual --pixelsize={args.pixelsize}"
    if args.cell_from_mat:
        return cell_from_mat(mat), "mat nominal_pixelsize"
    cell, bl = cell_from_ms(ms)
    return cell, f"MS longest baseline ({bl:.4e} lambda, n=2)"


def jybeam_to_jypixel(dirty_fits: Path, out_fits: Path) -> dict:
    """Convert Jy/beam -> Jy/pixel (same formula as
    src/utils/transforms.py::convert_jybeam_to_jypixel) and write out_fits."""
    from astropy.io import fits

    data, hdr = fits.getdata(dirty_fits, header=True)
    if "BMAJ" not in hdr or "BMIN" not in hdr:
        raise RuntimeError(f"{dirty_fits} has no BMAJ/BMIN; cannot convert to Jy/pixel")
    bmaj, bmin = hdr["BMAJ"], hdr["BMIN"]
    if bmaj <= 0 or bmin <= 0:
        raise RuntimeError(f"{dirty_fits} has BMAJ={bmaj}, BMIN={bmin}; cannot convert")
    factor = (4.0 * np.log(2.0) * hdr["CDELT1"] ** 2) / (np.pi * bmaj * bmin)
    data = data * factor
    hdr["BUNIT"] = "Jy/pixel"
    fits.writeto(out_fits, data, hdr, overwrite=True)
    return {"bmaj_deg": bmaj, "bmin_deg": bmin,
            "bpa_deg": hdr.get("BPA"), "beam_to_pix_factor": factor}


_CASA_DIRTY_SCRIPT = """
from casatasks import tclean, exportfits
tclean(vis={ms!r}, imagename={prefix!r}, cell={cell!r}, imsize={imsize},
       specmode="mfs", deconvolver="hogbom", gridder="standard",
       weighting="briggs", robust={robust}, niter=0, stokes="I",
       datacolumn="data")
exportfits(imagename={prefix!r} + ".image",
           fitsimage={fits_out!r}, overwrite=True)
"""


_CASA_CLEAN_SCRIPT = """
import glob, os, shutil, time
from casatasks import tclean, exportfits

common = dict(vis={ms!r}, cell={cell!r}, imsize={imsize}, specmode="mfs",
              deconvolver="hogbom", gridder="standard",
              weighting="briggs", robust={robust}, stokes="I",
              datacolumn="data")

# Dirty pass (niter=0) on a scratch name: its wall time estimates the
# gridding/setup share, same convention as src/algorithms/casa.py.
t0s = time.perf_counter()
tclean(imagename={prefix!r} + ".timing", niter=0, savemodel="none", **common)
t0e = time.perf_counter()
setup_time = t0e - t0s
for f in glob.glob({prefix!r} + ".timing*"):
    shutil.rmtree(f, ignore_errors=True) if os.path.isdir(f) else os.remove(f)

start = time.perf_counter()
tclean(imagename={prefix!r}, niter=1000000, gain=0.1,
       threshold={threshold!r}, savemodel="none", **common)
end = time.perf_counter()
duration = end - start
imaging_time = duration - setup_time
if imaging_time < 0:
    imaging_time = duration
print(f"Execution time: {{duration:.2f}} seconds")
print(f"Imaging time: {{imaging_time:.2f}} seconds")

exportfits(imagename={prefix!r} + ".image", fitsimage={fits_out!r},
           overwrite=True)
"""


def _remove_stale_casa_products(prefix: str) -> None:
    """tclean refuses to overwrite images whose coordinate system differs
    (e.g. from a previous run with a different cell); start clean."""
    import shutil
    for suffix in (".image", ".psf", ".model", ".sumwt", ".pb", ".mask",
                   ".residual", ".weight"):
        stale = Path(prefix + suffix)
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)
        elif stale.exists():
            stale.unlink()


def casa_dirty_pass(ms: Path, cell: float, imsize: int, robust: float,
                    work_dir: Path, dry_run: bool = False) -> dict:
    """Generate the dirty image with CASA tclean (niter=0) and export it.

    The exported FITS is in Jy/beam with the synthesized beam fitted by
    tclean in the header. Returns the log entry dict (empty on dry-run).
    """
    prefix = str(work_dir / "casa_dirty")
    fits_out = str(work_dir / "casa_dirty.dirty.fits")
    _remove_stale_casa_products(prefix)
    script = _CASA_DIRTY_SCRIPT.format(ms=str(ms), prefix=prefix, cell=f"{cell}arcsec",
                                       imsize=imsize, robust=robust, fits_out=fits_out)
    cmd = [str(CASA_ENV), "-c", script]
    if dry_run:
        print("[dirty] casa cmd: tclean niter=0 cell=%.6farcsec imsize=%d "
              "gridder=standard weighting=briggs robust=%s datacolumn=data" % (cell, imsize, robust))
        return {}
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not Path(fits_out).exists():
        raise SystemExit(f"CASA dirty pass failed (exit {res.returncode}):\n"
                         f"{res.stderr[-2000:]}\n{res.stdout[-2000:]}")
    return {"casa_dirty_jybeam": fits_out}


def cmd_dirty(args) -> None:
    ms, mat = target_paths(args.target)
    exp_dir = EXPERIMENTS_DIR / args.target
    exp_dir.mkdir(parents=True, exist_ok=True)
    work_dir = exp_dir / ".dirty_pass"
    cell, cell_source = resolve_cell(args, ms, mat)
    out_fits = exp_dir / f"{args.target}_dirty.fits"
    log_path = exp_dir / f"{args.target}_dirty.log"

    print("[dirty] target:", args.target)
    print("[dirty] ms:", ms)
    print("[dirty] cell: %.6f arcsec (%s)" % (cell, cell_source))
    print("[dirty] imsize: %d x %d" % (args.imsize, args.imsize))
    print("[dirty] weighting: briggs %.1f" % args.robust)
    print("[dirty] data column:", args.data_column)
    print("[dirty] tool: CASA tclean niter=0 (gridder=standard)")
    print("[dirty] output:", out_fits)
    if args.dry_run:
        casa_dirty_pass(ms, cell, args.imsize, args.robust, work_dir, dry_run=True)
        return

    work_dir.mkdir(parents=True, exist_ok=True)
    info = casa_dirty_pass(ms, cell, args.imsize, args.robust, work_dir)
    # Keep the dirty in Jy/beam (native CASA units): the sigma threshold
    # for casa/wsclean is measured from this image in Jy/beam directly.
    import shutil
    shutil.copy(info["casa_dirty_jybeam"], out_fits)
    from astropy.io import fits as _fits
    _hdr = _fits.getheader(out_fits)
    conv = {"bmaj_deg": _hdr["BMAJ"], "bmin_deg": _hdr["BMIN"],
            "bpa_deg": _hdr.get("BPA")}
    log = {
        "target": args.target,
        "mode": "dirty",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "ms": str(ms),
        "mat": str(mat),
        "tool": "casa tclean niter=0",
        "casa_cmd": f"tclean(vis, cell='{cell}arcsec', imsize={args.imsize}, "
                    f"specmode='mfs', deconvolver='hogbom', gridder='standard', "
                    f"weighting='briggs', robust={args.robust}, niter=0, stokes='I', "
                    f"datacolumn='{args.data_column.lower()}')",
        "cell_arcsec": cell,
        "cell_source": cell_source,
        "imsize": args.imsize,
        "weighting": f"briggs {args.robust}",
        "data_column": args.data_column,
        "units": {"output": "Jy/beam (native casa)", **info, **conv},
        "products": {
            "dirty_jypixel": str(out_fits),
            "casa_intermediates": str(work_dir),
        },
    }
    log_path.write_text(json.dumps(log, indent=2) + "\n")
    print("[dirty] done. log:", log_path)


def cmd_casa(args) -> None:
    if args.threshold is None:
        raise SystemExit("--casa requires --threshold <3*sigma in Jy/beam>")
    ms, mat = target_paths(args.target)
    exp_dir = EXPERIMENTS_DIR / args.target
    method_dir = exp_dir / f"{args.target}_casa"
    method_dir.mkdir(parents=True, exist_ok=True)
    cell, cell_source = resolve_cell(args, ms, mat)
    prefix = str(method_dir / f"{args.target}_casa")
    out_fits = method_dir / f"{args.target}_casa.fits"
    log_path = method_dir / f"{args.target}_casa.log"
    threshold_str = f"{args.threshold}Jy"

    print("[casa] target:", args.target)
    print("[casa] ms:", ms)
    print("[casa] cell: %.6f arcsec (%s)" % (cell, cell_source))
    print("[casa] imsize: %d x %d" % (args.imsize, args.imsize))
    print("[casa] weighting: briggs %.1f" % args.robust)
    print("[casa] deconvolver: hogbom | gridder: standard | gain: 0.1")
    print("[casa] threshold (absolute, 3*sigma): %s" % threshold_str)
    print("[casa] output:", out_fits)
    if args.dry_run:
        print("[casa] dry-run: not executing")
        return

    _remove_stale_casa_products(prefix)
    script = _CASA_CLEAN_SCRIPT.format(ms=str(ms), prefix=prefix, cell=f"{cell}arcsec",
                                       imsize=args.imsize, robust=args.robust,
                                       threshold=threshold_str, fits_out=str(out_fits))
    cmd = [str(CASA_ENV), "-c", script]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    exec_time_s = time.perf_counter() - t0
    if res.returncode != 0 or not out_fits.exists():
        raise SystemExit(f"CASA clean failed (exit {res.returncode}):\n"
                         f"{res.stderr[-2000:]}\n{res.stdout[-2000:]}")
    m = re.search(r"Imaging time: ([0-9.]+) seconds", res.stdout or "")
    if m is None:
        raise SystemExit("CASA script did not report 'Imaging time'")
    imaging_time_s = float(m.group(1))
    print("[casa] imaging time: %.1f s (clean - dirty-pass setup share, "
          "casa.py convention); execution time: %.1f s"
          % (imaging_time_s, exec_time_s))
    conv = jybeam_to_jypixel(out_fits, out_fits)
    log = {
        "target": args.target,
        "mode": "casa",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "imaging_time_s": round(imaging_time_s, 2),
        "execution_time_s": round(exec_time_s, 2),
        "imaging_time_note": ("clean tclean duration minus a same-grid niter=0 "
                              "dirty pass (setup share), same convention as "
                              "src/algorithms/casa.py"),
        "ms": str(ms),
        "tool": "casa tclean (hogbom)",
        "casa_cmd": f"tclean(vis, cell='{cell}arcsec', imsize={args.imsize}, "
                    f"specmode='mfs', deconvolver='hogbom', gridder='standard', "
                    f"weighting='briggs', robust={args.robust}, niter=1000000, "
                    f"gain=0.1, threshold='{threshold_str}', "
                    f"savemodel='none', stokes='I', datacolumn='data')",
        "threshold_jybeam": args.threshold,
        "threshold_note": "absolute threshold as passed via --threshold (3*sigma); "
                          "sigma measured by hand from the dirty image",
        "cell_arcsec": cell,
        "cell_source": cell_source,
        "imsize": args.imsize,
        "weighting": f"briggs {args.robust}",
        "data_column": args.data_column,
        "units": {"input": "Jy/beam (casa)", "output": "Jy/pixel", **conv},
        "products": {"image_jypixel": str(out_fits),
                     "casa_intermediates": str(method_dir)},
    }
    log_path.write_text(json.dumps(log, indent=2) + "\n")
    print("[casa] done. log:", log_path)


def cmd_wsclean(args) -> None:
    if args.threshold is None:
        raise SystemExit("--wsclean requires --threshold <3*sigma in Jy/beam>")
    ms, mat = target_paths(args.target)
    exp_dir = EXPERIMENTS_DIR / args.target
    method_dir = exp_dir / f"{args.target}_wsclean"
    method_dir.mkdir(parents=True, exist_ok=True)
    cell, cell_source = resolve_cell(args, ms, mat)
    prefix = method_dir / f"{args.target}_wsclean"
    out_fits = method_dir / f"{args.target}_wsclean.fits"
    log_path = method_dir / f"{args.target}_wsclean.log"

    # Single-scale clean (hogbom-equivalent), same gridding as the pipeline's
    # wsclean runs; -reorder is required by wsclean 3.7 for multi-spw MSs;
    # -no-update-model-required keeps the calibrated MS untouched.
    cmd = ["wsclean",
           "-size", str(args.imsize), str(args.imsize),
           "-scale", f"{cell}asec",
           "-padding", "2",
           "-weight", "briggs", str(args.robust),
           "-gridder", "wstacking", "-wstack-nwlayers", "1",
           "-gain", "0.1",
           "-abs-threshold", str(args.threshold),
           "-niter", "1000000",
           "-data-column", args.data_column,
           "-pol", "I", "-j", "1",
           "-no-update-model-required",
           "-reorder",
           "-name", str(prefix),
           str(ms)]

    print("[wsclean] target:", args.target)
    print("[wsclean] ms:", ms)
    print("[wsclean] cell: %.6f arcsec (%s)" % (cell, cell_source))
    print("[wsclean] imsize: %d x %d" % (args.imsize, args.imsize))
    print("[wsclean] weight: briggs %.1f | single-scale (hogbom-equiv) | gain 0.1" % args.robust)
    print("[wsclean] abs-threshold (3*sigma): %.6f Jy/beam" % args.threshold)
    print("[wsclean] cmd:", " ".join(cmd))
    print("[wsclean] output:", out_fits)
    if args.dry_run:
        print("[wsclean] dry-run: not executing")
        return

    for stale in method_dir.glob(f"{args.target}_wsclean-*.fits"):
        stale.unlink()
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    imaging_time_s = time.perf_counter() - t0
    image = Path(f"{prefix}-image.fits")
    if res.returncode != 0 or not image.exists():
        raise SystemExit(f"wsclean failed (exit {res.returncode}):\n"
                         f"{res.stderr[-2000:]}\n{res.stdout[-2000:]}")
    import shutil
    shutil.move(str(image), out_fits)
    conv = jybeam_to_jypixel(out_fits, out_fits)
    log = {
        "target": args.target,
        "mode": "wsclean",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "imaging_time_s": round(imaging_time_s, 2),
        "imaging_time_note": ("wsclean process wall time; run_wsclean.sh does "
                              "not split an internal imaging phase"),
        "ms": str(ms),
        "tool": "wsclean (single-scale clean)",
        "wsclean_cmd": cmd,
        "threshold_jybeam": args.threshold,
        "threshold_note": "absolute -abs-threshold as passed via --threshold "
                          "(3*sigma); sigma measured by hand from the dirty image",
        "cell_arcsec": cell,
        "cell_source": cell_source,
        "imsize": args.imsize,
        "weighting": f"briggs {args.robust}",
        "data_column": args.data_column,
        "units": {"input": "Jy/beam (wsclean)", "output": "Jy/pixel", **conv},
        "products": {"image_jypixel": str(out_fits),
                     "wsclean_intermediates": str(method_dir)},
    }
    log_path.write_text(json.dumps(log, indent=2) + "\n")
    print("[wsclean] done. log:", log_path)


def fix_fits_wcs(method_dir: Path, target: str, cell_arcsec: float) -> dict:
    """Fix CDELT (from the mat actually used) and refit the beam from the PSF.

    transforms.formatfits derives the pixel size from <MS_basename>.mat, which
    is wrong when the run used a different mat (e.g. the no-cutoff one).
    Keeps CRVAL/CRPIX written by formatfits (phase center is correct).
    """
    from astropy.io import fits
    from astropy.modeling import models, fitting

    cell_deg = cell_arcsec / 3600.0
    beam = {}
    psfs = sorted(method_dir.glob("*_psf.fits"))
    psf = psfs[0] if psfs else None
    if psf is not None and psf.exists():
        d = fits.getdata(psf).squeeze().astype(np.float64)
        ny, nx = d.shape
        half = max(8, min(ny, nx) // 4)
        cy, cx = ny // 2, nx // 2
        sub = d[cy - half:cy + half, cx - half:cx + half]
        yy, xx = np.mgrid[:sub.shape[0], :sub.shape[1]]
        g = models.Gaussian2D(amplitude=sub.max(), x_mean=half, y_mean=half,
                              x_stddev=3, y_stddev=3)
        g = fitting.LevMarLSQFitter()(g, xx, yy, sub)
        cd = abs(cell_deg)
        sx, sy = abs(g.x_stddev.value), abs(g.y_stddev.value)
        beam = {"BMAJ": float(2.3548 * max(sx, sy) * cd),
                "BMIN": float(2.3548 * min(sx, sy) * cd),
                "BPA": float(np.degrees(g.theta.value) % 180.0)}

    for f in method_dir.glob("*.fits"):
        data, hdr = fits.getdata(f, header=True)
        data = data.squeeze()
        ny, nx = data.shape
        hdr["CTYPE1"] = "RA---SIN"
        hdr["CRVAL1"] = 0.0
        hdr["CDELT1"] = -cell_deg
        hdr["CRPIX1"] = (nx / 2) + 0.5
        hdr["CUNIT1"] = "deg"
        hdr["CTYPE2"] = "DEC--SIN"
        hdr["CRVAL2"] = 0.0
        hdr["CDELT2"] = cell_deg
        hdr["CRPIX2"] = (ny / 2) + 0.5
        hdr["CUNIT2"] = "deg"
        hdr["BUNIT"] = "Jy/pixel"
        for k, v in beam.items():
            hdr[k] = v
        fits.writeto(f, data, hdr, overwrite=True)
    return {"cell_arcsec": cell_arcsec, **beam}


def cmd_r2d2(args) -> None:
    ms, mat = target_paths(args.target)
    exp_dir = EXPERIMENTS_DIR / args.target

    # Default: companion (cutoff) mat at the simulation grid, i.e. R2D2's
    # training scale. --nocut selects the full-baseline mat (probe, declared).
    mat_nocut = DATA_DIR / args.target / f"{args.target}_nocut.mat"
    if args.nocut:
        if not mat_nocut.exists():
            raise SystemExit(f"--nocut requested but not found: {mat_nocut}")
        data_file = mat_nocut
        cell_note = ("no uv_cutoff: full baselines to 4.47e6 lambda, "
                     "nominal_pixelsize 0.0231 arcsec (probe off training scale). "
                     "The mat holds ALL 24 SPWs but ONE channel per SPW "
                     "(freqid=0), both polarizations combined into one "
                     "pseudo-Stokes-I visibility per row, unflagged rows only "
                     "(786684 vis) — casa/wsclean mfs use every channel.")
    else:
        data_file = mat
        cell_note = ("uv_cutoff=2e6 mat at the simulation training grid, "
                     "nominal_pixelsize 0.0516 arcsec")
    method_dir = exp_dir / (f"{args.target}_r2d2_nocut" if args.nocut
                            else f"{args.target}_r2d2")

    r2d2_repo = BASE_DIR / "external" / "R2D2-RI"
    venv_python = BASE_DIR / ".envs" / ".r2d2" / "bin" / "python"
    ckpt = r2d2_repo / "ckpt" / "R2D2_small_ALMA3"
    cmd = [str(venv_python), "src/imager.py",
           "--config", str(BASE_DIR / "config" / "imaging" / "R2D2_small.yaml"),
           "--output_path", str(method_dir),
           "--data_file", str(data_file),
           "--save_all_outputs",
           "--architecture", "unet",
           "--prune",
           "--sigma_res_tol", "1e-4",
           "--super_resolution", str(args.super_res),
           "--meas_op_on_gpu",
           "--im_dim_x", str(args.imsize), "--im_dim_y", str(args.imsize),
           "--weight_robustness", str(args.robust),
           "--weight_type", "briggs",
           "--num_iter", "5",
           "--series", "R2D2",
           "--layers", "1",
           "--ckpt_path", str(ckpt),
           "--ckpt_realisations", "1",
           "--num_chans", "32"]

    from scipy.io import loadmat as _loadmat
    nominal_cell = float(_loadmat(data_file)["nominal_pixelsize"].item())
    print("[r2d2] target:", args.target)
    print("[r2d2] data file:", data_file)
    print("[r2d2] mat:", cell_note)
    print("[r2d2] im_dim: %d x %d | super_resolution: %s -> cell %.6f arcsec"
          % (args.imsize, args.imsize, args.super_res, nominal_cell / args.super_res))
    print("[r2d2] weights: briggs %.1f | num_iter: 5 (k5) | ckpt: R2D2_small_ALMA3" % args.robust)
    if args.nocut:
        print("[r2d2] NOTE: probe off training scale (network trained at ~0.05 arcsec/px)")
    print("[r2d2] output:", method_dir)
    if args.dry_run:
        print("[r2d2] cmd:", " ".join(cmd))
        print("[r2d2] dry-run: not executing")
        return

    import shutil
    if method_dir.exists():
        shutil.rmtree(method_dir)
    method_dir.mkdir(parents=True)
    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=r2d2_repo, capture_output=True, text=True)
    imaging_time_s = time.perf_counter() - t0
    log_txt = method_dir / f"{args.target}_r2d2_script.log"
    log_txt.write_text((res.stdout or "") + (res.stderr or "") +
                       f"\nImaging time: {imaging_time_s:.2f} seconds\n")
    if res.returncode != 0:
        raise SystemExit(f"r2d2 failed (exit {res.returncode}), see {log_txt}:\n"
                         f"{res.stderr[-2000:]}")

    # Products land in <method_dir>/<subdir>/; flatten like run_r2d2.sh.
    subdirs = [d for d in method_dir.iterdir() if d.is_dir()]
    model_image = None
    for d in subdirs:
        candidate = d / "R2D2_model_image.fits"
        if candidate.exists():
            model_image = candidate
            psf = d / "PSF.fits"
            break
    if model_image is None:
        raise SystemExit(f"R2D2_model_image.fits not found under {method_dir}, see {log_txt}")
    import shutil
    out_fits = method_dir / f"{args.target}_r2d2.fits"
    shutil.move(str(model_image), out_fits)
    if psf.exists():
        shutil.move(str(psf), method_dir / f"{args.target}_r2d2_psf.fits")
    for d in subdirs:
        for f in d.iterdir():
            shutil.move(str(f), method_dir / f.name)
        d.rmdir()

    # Pixel size from the mat actually used (nominal / super_resolution);
    # formatfits would take the companion <target>.mat (cutoff) scale.
    cell_used = nominal_cell / args.super_res
    beam_info = fix_fits_wcs(method_dir, args.target, cell_used)
    from astropy.io import fits as _fits
    _hdr = _fits.getheader(out_fits)
    beam_info = {"bmaj_deg": _hdr.get("BMAJ"), "bmin_deg": _hdr.get("BMIN"),
                 "bpa_deg": _hdr.get("BPA"), **beam_info}

    log = {
        "target": args.target,
        "mode": "r2d2_nocut" if args.nocut else "r2d2",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "ms": str(ms),
        "data_file": str(data_file),
        "mat_note": cell_note,
        "tool": "R2D2-RI imager.py" + (" (probe off training scale, declared)" if args.nocut else ""),
        "cmd": cmd,
        "im_dim": args.imsize,
        "super_resolution": args.super_res,
        "cell_arcsec": cell_used,
        "weighting": f"briggs {args.robust}",
        "num_iter": 5,
        "ckpt": str(ckpt),
        "imaging_time_s": round(imaging_time_s, 2),
        "imaging_time_note": ("imager.py process wall time; run_r2d2.sh does "
                              "not split an internal imaging phase"),
        "units": {"output": "Jy/pixel", **beam_info},
        "products": {"image": str(out_fits), "dir": str(method_dir)},
    }
    log_path = method_dir / f"{args.target}_r2d2.log"
    log_path.write_text(json.dumps(log, indent=2) + "\n")
    print("[r2d2] done. log:", log_path)


def make_rescaled_mat(src_mat: Path, dst_mat: Path, cell_arcsec: float) -> Path:
    """Copy a mat with nominal_pixelsize overwritten (same trick as r2d2's
    --super-resolution: uv in lambda are unchanged, only the image domain is
    rescaled). deepinverse.py derives the operator cell from this value."""
    from scipy.io import loadmat, savemat

    d = loadmat(src_mat)
    d["nominal_pixelsize"] = np.full((1, 1), float(cell_arcsec))
    savemat(dst_mat, d)
    return dst_mat


def deepinv_cell_from_r2d2(exp_dir: Path, target: str, nocut_mat: Path) -> tuple[float, str]:
    """Cell matching the accepted r2d2 run (same grid for both regularized
    methods); falls back to nocut_nominal / 0.4847 (the r2d2 --super-res used)."""
    from astropy.io import fits

    r2d2_fits = exp_dir / f"{target}_r2d2_nocut" / f"{target}_r2d2.fits"
    if r2d2_fits.exists():
        cell = abs(float(fits.getheader(r2d2_fits)["CDELT2"])) * 3600.0
        return cell, f"accepted r2d2 grid ({r2d2_fits.name} CDELT2)"
    return cell_from_mat(nocut_mat) / 0.4847, "nocut nominal_pixelsize / 0.4847 (r2d2 super-res)"


def cmd_deepinv(args) -> None:
    ms, mat = target_paths(args.target)
    exp_dir = EXPERIMENTS_DIR / args.target
    method_dir = exp_dir / f"{args.target}_deepinverse"
    if args.tag:
        method_dir = method_dir.with_name(method_dir.name + f"_{args.tag}")
    nocut_mat = DATA_DIR / args.target / f"{args.target}_nocut.mat"
    imsize = 64 if args.imsize == DEFAULT_IMSIZE else args.imsize
    cell, cell_source = deepinv_cell_from_r2d2(exp_dir, args.target, nocut_mat)
    rescaled_mat = exp_dir / f"{args.target}_deepinv.mat"
    out_fits = method_dir / f"{args.target}_deepinverse.fits"
    log_path = method_dir / f"{args.target}_deepinverse.log"

    # l1_i50_v2 from the archival yaml (single source of truth with the sweep).
    parse_cmd = [sys.executable, str(BASE_DIR / "scripts" / "parse_target_yaml.py"),
                 "--target", args.target, "--method", "deepinv"]
    cfg = json.loads(subprocess.run(parse_cmd, capture_output=True, text=True,
                                    cwd=BASE_DIR, check=True).stdout)
    cfg.pop("weight_robustness", None)  # only used by generate_weights.sh
    if args.lambda_factor is not None:
        cfg["lambda_factor"] = args.lambda_factor
    cfg["img_size"] = [imsize, imsize]
    cfg["vis_path"] = str(rescaled_mat)
    cfg["ms_path"] = str(ms)
    cfg["out_dir"] = str(method_dir) + "/"

    print("[deepinv] target:", args.target)
    print("[deepinv] ms:", ms)
    print("[deepinv] nocut mat:", nocut_mat)
    print("[deepinv] cell: %.6f arcsec (%s)" % (cell, cell_source))
    print("[deepinv] imsize: %d x %d (FOV %.2f arcsec)" % (imsize, imsize, imsize * cell))
    print("[deepinv] config: l1_i50_v2 (yaml deepinv section) + fluxcal, vis_path/out_dir overridden")
    print("[deepinv] lambda_factor: %s" % cfg["lambda_factor"])
    print("[deepinv] output:", out_fits)
    if args.dry_run:
        print("[deepinv] dry-run: not executing")
        return

    # Flux calibration computed here (benchmark env casacore) and passed
    # explicitly: reading this 2.4M-row MS with casacore inside the .deepinv
    # env aborts with a double-free corruption.
    from src.utils.flux_estimate import estimate_total_flux
    f_est, f_est_path = estimate_total_flux(str(ms))
    print("[deepinv] flux estimate: F_est=%.4f Jy (estimator=%s, computed in benchmark env)"
          % (f_est, f_est_path))
    cfg["flux_estimate"] = f_est

    # 1) weighting: natural only (mat nW), briggs=1. Deliberate deviation,
    #    declared in the log: wsclean 3.7 cannot persist briggs imaging weights
    #    on this multi-spw MS (-store-imaging-weights is lost with -reorder),
    #    so casa/wsclean's briggs robust=2 is not applied for deepinv.
    print("[deepinv] weighting: natural (mat nW), briggs=1 (declared deviation; "
          "casa/wsclean use briggs robust=2)")

    # 2) rescaled copy of the mat: cell of the accepted r2d2 grid (declared).
    make_rescaled_mat(nocut_mat, rescaled_mat, cell)

    # 3) run deepinverse (clears out_dir contents itself).
    method_dir.mkdir(parents=True, exist_ok=True)
    venv_python = BASE_DIR / ".envs" / ".deepinv" / "bin" / "python"
    env = dict(os.environ, PYTHONPATH=str(BASE_DIR))
    cmd = [str(venv_python), "src/algorithms/deepinverse.py",
           "--target", args.target, "--config", json.dumps(cfg), "--nval", "1"]
    print("[deepinv] cmd:", " ".join(cmd[:4]), "--config '<l1_i50_v2 + overrides>' --nval 1")
    res = subprocess.run(cmd, cwd=BASE_DIR, env=env, capture_output=True, text=True)
    script_log = method_dir / f"{args.target}_deepinverse_script.log"
    script_log.write_text((res.stdout or "") + (res.stderr or ""))
    if res.returncode != 0 or not out_fits.exists():
        raise SystemExit(f"deepinv failed (exit {res.returncode}), see {script_log}:\n"
                         f"{res.stderr[-2000:]}")
    m = re.search(r"Imaging time: ([0-9.]+) seconds", res.stdout or "")
    if m is None:
        raise SystemExit("deepinverse.py did not report 'Imaging time'")
    imaging_time_s = float(m.group(1))
    print("[deepinv] imaging time: %.1f s (FISTA solve only, internal timing "
          "of deepinverse.py)" % imaging_time_s)

    # 4) WCS: savefits writes CDELT from the rescaled mat; refit beam from PSF.
    beam_info = fix_fits_wcs(method_dir, args.target, cell)
    from astropy.io import fits as _fits
    _hdr = _fits.getheader(out_fits)
    beam_info = {"bmaj_deg": _hdr.get("BMAJ"), "bmin_deg": _hdr.get("BMIN"),
                 "bpa_deg": _hdr.get("BPA"), **beam_info}

    log = {
        "target": args.target,
        "mode": "deepinv",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "ms": str(ms),
        "data_file": str(rescaled_mat),
        "mat_note": ("rescaled copy of the no-cutoff mat: nominal_pixelsize set to "
                     "the accepted r2d2 cell; uv unchanged (image-domain rescale, "
                     "declared). Weighting: natural only (mat nW, sqrt of MS "
                     "natural weights); briggs=1 — declared deviation, casa/wsclean "
                     "use briggs robust=2 (wsclean 3.7 cannot persist imaging "
                     "weights on this multi-spw MS). The mat holds ALL 24 SPWs "
                     "but ONE channel per SPW (freqid=0), both polarizations "
                     "combined into one pseudo-Stokes-I visibility per row, "
                     "unflagged rows only (786684 vis) — casa/wsclean mfs use "
                     "every channel of every SPW."),
        "tool": "deepinverse.py (FISTA + WaveletPrior, l1_i50_v2 + fluxcal)",
        "cell_arcsec": cell,
        "cell_source": cell_source,
        "imsize": imsize,
        "lambda_factor": cfg["lambda_factor"],
        "tag": args.tag,
        "imaging_time_s": round(imaging_time_s, 2),
        "imaging_time_note": ("FISTA solve only (deepinverse.py internal timing, "
                              "cuda-synchronized); excludes data loading and "
                              "operator/opnorm setup"),
        "flux_estimate_jy": f_est,
        "flux_estimator": f_est_path,
        "fluxcal_note": ("F_est (short-baseline extrapolation to uv=0) computed by "
                         "the orchestrator in the benchmark env and passed via config; "
                         "deepinverse scales the reconstruction so its total flux "
                         "equals F_est."),
        "config": {k: v for k, v in cfg.items() if k not in ("vis_path", "out_dir", "ms_path")},
        "units": {"output": "Jy/pixel", **beam_info},
        "products": {"image_jypixel": str(out_fits), "dir": str(method_dir)},
    }
    log_path.write_text(json.dumps(log, indent=2) + "\n")
    print("[deepinv] done. log:", log_path)


def _panel_load(path: Path, crop_fov_arcsec: float | None = None):
    """Load a panel FITS: squeezed float64, cell (arcsec), beam in pixels."""
    from astropy.io import fits

    d, h = fits.getdata(path, header=True)
    d = np.squeeze(d).astype(np.float64)
    cell = abs(float(h["CDELT1"])) * 3600.0
    if crop_fov_arcsec is not None:
        n = int(round(crop_fov_arcsec / cell))
        n = min(n, min(d.shape))
        c = d.shape[0] // 2
        d = d[c - n // 2: c - n // 2 + n, c - n // 2: c - n // 2 + n]
    beam = None
    if "BMAJ" in h and "BMIN" in h and h["BMAJ"] > 0 and h["BMIN"] > 0:
        beam = (float(h["BMAJ"]) * 3600.0 / cell,
                float(h["BMIN"]) * 3600.0 / cell,
                float(h.get("BPA", 0.0)))
    return d, cell, beam


def _corner_rms(img, patch_size: int = 9) -> tuple[float, int]:
    """Corner-patch noise sigma, the user's estimator (src/utils/dirty_noise.py):
    four square corner patches, a 2D plane fit and removed per patch, patches
    with std > 2x the median std discarded, sigma = mean std of the survivors.
    Returns (sigma, n_valid_patches) in the image's own units."""
    from src.utils.dirty_noise import (compute_patch_stats,
                                       define_corner_patches,
                                       filter_patches_by_std_consistency)

    img = np.asarray(img, dtype=np.float64)
    patches = define_corner_patches(img.shape, patch_size=patch_size)
    stats = [compute_patch_stats(img, p, detrend=True) for p in patches]
    valid, _ = filter_patches_by_std_consistency(stats)
    if not valid:
        raise RuntimeError("corner-patch noise: all 4 corner patches discarded")
    return float(np.mean([s.std for s in valid])), len(valid)


def _method_imaging_time(fits_path: Path) -> float | None:
    """imaging_time_s from the method's sibling JSON log (<stem>.log)."""
    log_path = fits_path.parent / (fits_path.stem + ".log")
    if not log_path.exists():
        return None
    try:
        return float(json.loads(log_path.read_text())["imaging_time_s"])
    except (ValueError, KeyError):
        return None


def _fmt_duration(s: float) -> str:
    s = int(round(s))
    return f"{s}s" if s < 90 else f"{s // 60}m {s % 60:02d}s"


def cmd_png(args) -> None:
    """Render the archival comparison panel from existing products (no recon).

    Mold: upper row of the noisevar comparisons (render_all_preview, compact
    geometry, own colorbar per panel, log-count histogram under each panel).
    Row: DSHARP (published reference, Jy/beam -> Jy/pixel) | casa | wsclean |
    deepinv | r2d2. No GT and no error row; each reconstruction title reports
    the corner-patch rms and dynamic range (peak/rms) of that image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from src.utils.studies.noisevar.plot_noisevar_comparisons import (
        HIST_BINS, CB_H, CB_PAD, add_colorbar, hist_peak, style_hist)

    exp_dir = EXPERIMENTS_DIR / args.target
    if args.units == "jyas2":
        unit = "Jy/arcsec²"
        out_png = BASE_DIR / "results" / "archival" / f"{args.target}_panel_jyas2.png"
    else:
        unit = "mJy/px"
        out_png = BASE_DIR / "results" / "archival" / f"{args.target}_panel.png"
    deepinv_dir = exp_dir / (f"{args.target}_deepinverse_{args.tag}" if args.tag
                             else f"{args.target}_deepinverse")
    r2d2_dir = exp_dir / f"{args.target}_r2d2_nocut"  # accepted no-cutoff run

    # DSHARP published reference: crop ~3.1 arcsec FOV. The reference FITS
    # sits at data/archival/<target>.fits (next to the target directory, not
    # inside it). Its native unit is Jy/beam.
    dsharp_path = DATA_DIR / f"{args.target}.fits"
    dsharp, _, beam_ds = _panel_load(dsharp_path, crop_fov_arcsec=3.1)
    from astropy.io import fits as _fits
    hdr = _fits.getheader(dsharp_path)
    factor = (4.0 * np.log(2.0) * hdr["CDELT1"] ** 2) / (np.pi * hdr["BMAJ"] * hdr["BMIN"])
    dsharp_jypix = dsharp * factor  # Jy/beam -> Jy/pixel (flux-sum basis)
    if args.units == "jyas2":
        # Jy/beam -> Jy/arcsec^2 dividing by the Gaussian beam area.
        beam_area = (np.pi * (hdr["BMAJ"] * 3600.0) * (hdr["BMIN"] * 3600.0)
                     / (4.0 * np.log(2.0)))
        dsharp = dsharp / beam_area
    else:
        dsharp = dsharp_jypix * 1e3  # -> mJy/pixel

    # Reference metrics, same corner-patch estimator as the reconstructions.
    # Its flux is the sum over the 3.1 arcsec crop - which matches the
    # reconstructions' FOV (64 x 0.0476" = 3.05"), so the comparison is fair.
    rms_ds, n_ds = _corner_rms(dsharp)
    dr_ds = float(dsharp.max()) / rms_ds
    flux_ds = float(dsharp_jypix.sum())
    sub_ds = (f"rms={rms_ds:.3g} {unit} | DR={dr_ds:.4g}\n"
              f"flux={flux_ds:.3g} Jy")
    print("[png]   DSHARP: rms=%.4g %s (%d/4 corner patches), peak=%.4g, DR=%.4g, "
          "flux=%.4g Jy" % (rms_ds, unit, n_ds, dsharp.max(), dr_ds, flux_ds))
    panels = [("DSHARP (referencia publicada)", sub_ds, dsharp, beam_ds)]
    for title, path, beam_override in [
        ("casa", exp_dir / f"{args.target}_casa" / f"{args.target}_casa.fits", None),
        ("wsclean", exp_dir / f"{args.target}_wsclean" / f"{args.target}_wsclean.fits", None),
        ("BPSA", deepinv_dir / f"{args.target}_deepinverse.fits", "none"),
        ("R2Disk2", r2d2_dir / f"{args.target}_r2d2.fits", "none"),
    ]:
        img, cell, beam = _panel_load(path)
        if beam_override == "none":
            beam = None  # super-res grids: no clean beam to draw
        if args.units == "jyas2":
            q = img / cell ** 2  # Jy/pixel -> Jy/arcsec^2
        else:
            q = img * 1e3  # Jy/pixel -> mJy/pixel
        flux = float(img.sum())  # Jy/pixel sum = total flux (unit-independent)
        rms, n_patches = _corner_rms(q)
        dr = float(q.max()) / rms
        t_img = _method_imaging_time(path)
        t_txt = f" | t={_fmt_duration(t_img)}" if t_img is not None else ""
        sub = (f"rms={rms:.3g} {unit} | DR={dr:.4g}\n"
               f"flux={flux:.3g} Jy{t_txt}")
        print("[png]   %s: rms=%.4g %s (%d/4 corner patches), peak=%.4g, "
              "DR=%.4g, flux=%.4g Jy, imaging t=%s"
              % (title, rms, unit, n_patches, q.max(), dr, flux,
                 _fmt_duration(t_img) if t_img is not None else "n/a"))
        panels.append((title, sub, q, beam))

    print("[png] target:", args.target)
    for title, sub, img, beam in panels:
        print("[png]   %s: %s, beam=%s" % (title.split(" (")[0], img.shape,
                                           "yes" if beam else "no"))
    print("[png] output:", out_png)
    if args.dry_run:
        print("[png] dry-run: not executing")
        return

    # Shared log-count y limits across all panels (mold: across the row).
    peak_ct = 1
    for _, _, img, _ in panels:
        peak_ct = max(peak_ct, hist_peak(img, float(img.min()), float(img.max())))
    hist_ylim = (0.8, peak_ct * 1.2)

    fig = plt.figure(figsize=(13.33, 5.5), dpi=150)
    outer = fig.add_gridspec(1, 5, left=0.045, right=1 - 0.045, top=0.82,
                             bottom=0.115, wspace=0.012)
    imshow_kw = dict(origin="lower", interpolation="nearest", aspect="equal")
    hist_pairs, top_panels = [], []

    for col, (title, sub, img, beam) in enumerate(panels):
        cell = outer[0, col].subgridspec(2, 1, height_ratios=[3.0, 0.85],
                                         hspace=0.5)
        ax_img = fig.add_subplot(cell[0])
        v0, v1 = float(img.min()), float(img.max())
        im = ax_img.imshow(img, vmin=v0, vmax=v1, cmap="inferno", **imshow_kw)
        ax_img.set_title(title if sub is None else f"{title}\n{sub}",
                         fontsize=9.5, pad=6)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_anchor("N")
        if beam is not None:
            bmaj_px, bmin_px, bpa = beam
            for ec, lw in (("black", 2.4), ("white", 1.2)):
                ax_img.add_patch(Ellipse(
                    (0.11 * img.shape[1], 0.11 * img.shape[0]),
                    width=bmin_px, height=bmaj_px, angle=90.0 + bpa,
                    fill=False, ec=ec, lw=lw, zorder=5))
        ax_hist = fig.add_subplot(cell[1])
        style_hist(ax_hist, img, np.linspace(v0, v1, HIST_BINS + 1),
                   ylim=hist_ylim, ylabels=(col == 0))
        ax_hist.set_xlabel(f"Value ({unit})", fontsize=8, color="0.25")
        if col == 0:
            ax_hist.set_ylabel("count (log)", fontsize=8, color="0.25",
                               labelpad=2)
        hist_pairs.append((ax_img, ax_hist))
        top_panels.append((ax_img, im))

    # Fit each histogram right below its colorbar, leaving room for the
    # colorbar tick labels (mold's align step + label height).
    fig.canvas.draw()
    offset = CB_PAD + CB_H + 0.05
    for ax_img, ax_hist in hist_pairs:
        p_img = ax_img.get_position()
        p_hist = ax_hist.get_position()
        ax_hist.set_position(
            [p_img.x0, p_img.y0 - offset - p_hist.height,
             p_img.width, p_hist.height])

    for ax, im in top_panels:
        add_colorbar(fig, im, ax)

    fig.suptitle(f"{args.target} — datos de archivo (sin ground truth) — "
                 f"todas en {unit}\n"
                 "rms por parches de esquina (4 esquinas, plane-detrend, "
                 "estimador del sweep); DR = pico/rms; "
                 "t = tiempo de imaging (casa: clean - dirty pass; deepinv: "
                 "FISTA interno; wsclean/r2d2: proceso)",
                 fontsize=10, y=0.996)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    print("[png] done:", out_png)


def cmd_lam_sweep(args) -> None:
    """Render the deepinv lambda-sweep strip from existing products (no recon).

    One panel per lambda variant (dir <target>_deepinverse[_lam<tag>]) plus
    the accepted r2d2 run as reference. Shared vmax, native Jy/pixel. Each
    title reports peak, total flux, and the corner-patch rms + DR (the user's
    estimator from src/utils/dirty_noise.py, same as --png). The canonical
    <target>_deepinverse dir holds the accepted lambda (1e-3); the 1e-1
    original lives in the _lam1e-1 dir.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exp_dir = EXPERIMENTS_DIR / args.target
    out_png = BASE_DIR / "results" / "archival" / f"{args.target}_deepinv_lambda_sweep.png"
    t = args.target
    entries = [
        ("λ=1e-1", exp_dir / f"{t}_deepinverse_lam1e-1", f"{t}_deepinverse.fits"),
        ("λ=3e-2", exp_dir / f"{t}_deepinverse_lam3e-2", f"{t}_deepinverse.fits"),
        ("λ=1e-2", exp_dir / f"{t}_deepinverse_lam1e-2", f"{t}_deepinverse.fits"),
        ("λ=3e-3", exp_dir / f"{t}_deepinverse_lam3e-3", f"{t}_deepinverse.fits"),
        ("λ=1e-3", exp_dir / f"{t}_deepinverse", f"{t}_deepinverse.fits"),
        ("λ=3e-4", exp_dir / f"{t}_deepinverse_lam3e-4", f"{t}_deepinverse.fits"),
        ("r2d2 k5 (referencia)", exp_dir / f"{t}_r2d2_nocut", f"{t}_r2d2.fits"),
    ]

    panels = []
    for label, method_dir, fits_name in entries:
        path = method_dir / fits_name
        if not path.exists():
            print("[lam-sweep] WARNING: missing %s, skipping" % path)
            continue
        img, _, _ = _panel_load(path)
        img_mjy = img * 1e3  # Jy/pixel -> mJy/pixel
        rms, n_patches = _corner_rms(img_mjy)
        dr = float(img_mjy.max()) / rms
        t_img = _method_imaging_time(path)
        t_txt = f" | t={_fmt_duration(t_img)}" if t_img is not None else ""
        title = (f"{label}\npeak={img_mjy.max():.1f} mJy/px, flux={img.sum():.3f} Jy\n"
                 f"rms={rms:.3g} mJy/px | DR={dr:.4g}{t_txt}")
        print("[lam-sweep] %s: rms=%.4g mJy/px (%d/4 corner patches), DR=%.4g"
              % (label, rms, n_patches, dr))
        panels.append((title, img))

    print("[lam-sweep] output:", out_png)
    if args.dry_run:
        print("[lam-sweep] dry-run: not executing")
        return
    if not panels:
        raise SystemExit("[lam-sweep] no variant products found")

    vmax = max(img.max() for _, img in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 4.8))
    axes = np.atleast_1d(axes)
    for ax, (title, img) in zip(axes, panels):
        im = ax.imshow(img, cmap="inferno", origin="lower", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=list(axes), shrink=0.8, label="Jy/pixel")
    fig.suptitle(f"{t} archival — DeepInv lambda sweep (l1_i50_v2 resto fijo) "
                 "@ 64x64, 0.0476\"/px — no-cutoff mat, natural weighting "
                 "(briggs=1, declared), fluxcal F_est=0.744 Jy. "
                 "rms/DR: parches de esquina, DR = pico/rms", fontsize=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[lam-sweep] done:", out_png)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True, help="archival target name (data/archival/<target>)")
    p.add_argument("--dirty", action="store_true", help="generate only the full-resolution dirty image")
    p.add_argument("--casa", action="store_true", help="CASA tclean reconstruction (hogbom, 3*sigma absolute threshold)")
    p.add_argument("--wsclean", action="store_true", help="WSClean reconstruction (single-scale, 3*sigma absolute threshold)")
    p.add_argument("--r2d2", action="store_true", help="R2D2 (k5) reconstruction, imsize x imsize pixels (default: cutoff mat; --nocut for full baselines)")
    p.add_argument("--deepinv", action="store_true", help="DeepInverse (l1_i50_v2 + fluxcal) on the no-cutoff mat at the accepted r2d2 grid (default 64x64)")
    p.add_argument("--png", action="store_true", help="render only the comparison panel PNG from existing products (no reconstruction)")
    p.add_argument("--lambda-factor", type=float, default=None,
                   help="with --deepinv: override lambda_factor (lower = more data fidelity/detail)")
    p.add_argument("--tag", default=None,
                   help="with --deepinv: output dir suffix (<target>_deepinverse_<tag>) for parameter probes")
    p.add_argument("--units", choices=("mjypx", "jyas2"), default="mjypx",
                   help="with --png: display units - mJy/px per image pixel or "
                        "Jy/arcsec^2 surface brightness (Jy/pixel / cell^2); "
                        "the DSHARP reference converts via its beam area")
    p.add_argument("--lam-sweep", action="store_true",
                   help="render the deepinv lambda-variant strip (existing products, "
                        "no recon) with peak, flux, corner-patch rms and DR per panel")
    p.add_argument("--nocut", action="store_true", help="with --r2d2: use the full-baseline <target>_nocut.mat instead of the cutoff mat")
    p.add_argument("--super-res", type=float, default=1.0,
                   help="with --r2d2: super-resolution factor, cell = nominal_pixelsize / factor "
                        "(default: %(default)s; 0.4847 = training scale 0.0476 arcsec on the nocut mat)")
    p.add_argument("--threshold", type=float, default=None,
                   help="absolute threshold in Jy/beam, already multiplied by 3 "
                        "(e.g. rms 1.515e-3 -> pass 4.545e-3); required with --casa")
    p.add_argument("--imsize", type=int, default=DEFAULT_IMSIZE,
                   help="image size in pixels (default: %(default)s for --dirty, 64 for --deepinv)")
    p.add_argument("--robust", type=float, default=DEFAULT_ROBUST,
                   help="Briggs robust parameter (default: %(default)s)")
    p.add_argument("--data-column", default=DEFAULT_DATA_COLUMN,
                   help="MS data column (default: %(default)s)")
    p.add_argument("--pixelsize", "--cell", dest="pixelsize", type=float, default=None,
                   help="pixel size in arcsec (default: from MS longest baseline)")
    p.add_argument("--cell-from-mat", action="store_true",
                   help="use nominal_pixelsize from the .mat instead of the MS")
    p.add_argument("--dry-run", action="store_true", help="print what would run, execute nothing")
    args = p.parse_args()

    if not args.dirty and not args.casa and not args.wsclean and not args.r2d2 and not args.deepinv and not args.png and not args.lam_sweep:
        p.error("no mode selected (available: --dirty, --casa, --wsclean, --r2d2, --deepinv, --png, --lam-sweep)")
    if args.dirty:
        cmd_dirty(args)
    if args.casa:
        cmd_casa(args)
    if args.wsclean:
        cmd_wsclean(args)
    if args.r2d2:
        cmd_r2d2(args)
    if args.deepinv:
        cmd_deepinv(args)
    if args.png:
        cmd_png(args)
    if args.lam_sweep:
        cmd_lam_sweep(args)


if __name__ == "__main__":
    main()
