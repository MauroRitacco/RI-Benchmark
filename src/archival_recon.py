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
            goes to experiments/archival/<target>/<target>_casa/<target>_casa/
            with a log.

  --wsclean WSClean reconstruction (single-scale clean, hogbom-equivalent,
            same grid + threshold convention as --casa). Output (Jy/pixel)
            goes to experiments/archival/<target>/<target>_wsclean/
            <target>_wsclean/ with a log.

  --deepinv DeepInverse reconstruction (FISTA + WaveletPrior, archival-yaml
            config l1_i50_v2 + fluxcal) on the no-cutoff mat at the accepted
            r2d2 grid (64x64 @ ~0.0476 arcsec): the mat is copied with
            nominal_pixelsize rescaled (uv unchanged; declared). Weighting is
            natural only (mat nW), briggs=1 — declared deviation: wsclean 3.7
            cannot persist briggs imaging weights on this multi-spw MS, while
            casa/wsclean image with briggs robust=2. Output (Jy/pixel) goes
            to experiments/archival/<target>/<target>_deepinverse/
            <target>_deepinverse[_<variant>]/ with a log. Variants (--allch,
            --tag, e.g. sweep lambdas) are subdirs of the method container;
            run_dir() builds all paths.

  --tidy    Render the minimal abstract figure (no reconstruction) from the
            existing products: DSHARP reference + casa + wsclean (canonical
            robust 2 runs) + BPSA (avg_lam3e-4) + R2Disk2 (avg). Everything
            is converted to surface brightness in mJy/arcsec^2 (DSHARP from
            Jy/beam via its header beam, 1.133 x BMAJ x BMIN; the Jy/pixel
            reconstructions via their own pixel area), shown on a single
            common asinh scale (vmin=0, common vmax, per-method asinh
            linear_width from the --lw-<method> flags in mJy/px of the
            64x64 grid, the DSHARP panel from 10 x its corner rms), single
            colorbar, no histograms / DR / rms / timings. Beam drawn on
            DSHARP/tclean/WSClean only (BPSA and R2Disk2 are model images,
            not convolved with a clean beam); common 0.5 arcsec angular
            scale bar per panel. Saved to
            results/archival/<target>/<variant>/tidy/<target>_continuum_comparison.png
            (300 dpi) + .pdf.

  --png     Render only the comparison panel PNG (no reconstruction) from the
            existing products: DSHARP published reference + casa + wsclean +
            deepinv + r2d2. Faithful single-row port of the noisevar mold
            (src/utils/studies/noisevar/plot_noisevar_comparisons.py,
            render_all_preview, compact geometry, own colorbar per panel,
            log-count histogram under each panel). Saved to
            results/archival/<target>/<variant>/<target>_panel.png, where
            <variant> is v1 (1-channel mat), avg (channel-averaged mat) or
            all_channels (full-band mat, --allch).
            All images Jy/pixel (DSHARP
            converted from Jy/beam with its header beam). No GT, no error
            row, no metrics.

Run with the benchmark env (casacore + astropy + scipy):

    .envs/.benchmark/bin/python src/archival_recon.py --target <name> --dirty

Units convention (same as the rest of the pipeline): reconstruction FITS are
in Jy/pixel. The dirty image is kept in Jy/beam (native CASA units) because
the absolute sigma threshold for casa/wsclean is set from its rms: sigma
[Jy/beam] -> 3*sigma threshold, no conversion needed. Reconstructions are
converted with the *effective* beam area: 1/A_eff, A_eff from the dirty-PSF
image sum (peak-normalised). The residual/restored are in Jy/beam of the
dirty beam, whose effective area exceeds the fitted Gaussian (robust=2);
converting with the fitted-beam formula inflates extended emission by
A_eff/A_fit (~1.1-2.4x on archival data).
"""

from __future__ import annotations

import argparse
import json
import math
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

# Grid shared by all four methods on the archival targets (casa/wsclean
# --pixelsize; the r2d2/deepinv mats are rescaled copies at this cell).
COMMON_CELL = 0.047602

# asinh color scale (--png --asinh): linear_width = ASINH_LW_FAC x sigma,
# common to all reconstruction panels (sigma = the target's full-FOV dirty in
# display units). The DSHARP reference panel uses its own much quieter
# corner-patch rms instead: with the dirty-based width its faint ring sits on
# the colormap floor.
ASINH_LW_FAC = 10.0

# Same linear-width factor, but for the --lam-sweep strips (kept separate so
# the sweep stretch can be tuned independently of the comparison panels).
ASINH_LW_FAC_SWEEP = 3.5


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


def run_dir(exp_dir: Path, target: str, method: str,
            variant: str | None = None) -> Path:
    """Output dir of one reconstruction run, two fixed levels by method:

        experiments/archival/<target>/<target>_<method>/<target>_<method>[_<variant>]/

    The run dir holds files named <target>_<method>.* (identical names in
    every variant); the method-level dir is just a container that groups
    the canonical run with its variants (e.g. sweep lambdas, _allch).
    """
    name = f"{target}_{method}"
    return exp_dir / name / (f"{name}_{variant}" if variant else name)


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


def jybeam_to_jypixel(restored_fits: Path, out_fits: Path, model_fits: Path,
                      residual_fits: Path, psf_fits: Path) -> dict:
    """Rebuild the restored image in Jy/pixel from its products and write it.

    The restored = model (Jy/px, point sources) convolved with the clean
    beam (Gaussian, fitted BMAJ/BMIN) plus the residual, and the residual is
    in Jy/beam of the *dirty* beam, whose effective area A_eff exceeds the
    fitted Gaussian at robust=2. A single conversion factor on the restored
    therefore cannot be flux-exact: the model component converts with
    A_px/A_fit, the residual with 1/A_eff (A_eff from the PSF image sum,
    peak-normalised). This rebuilds the image term by term:

        out = model (*) clean_beam            [Jy/px, flux = F_model]
            + residual * (PSF_peak/PSF_sum)   [Jy/px, flux = F_residual]

    so the pixel sum equals F_model + F_residual exactly.
    """
    from astropy.io import fits
    from scipy.signal import fftconvolve

    hdr = fits.getheader(restored_fits)
    m = fits.getdata(model_fits).squeeze().astype(np.float64)
    r = fits.getdata(residual_fits).squeeze().astype(np.float64)
    psf = fits.getdata(psf_fits).squeeze().astype(np.float64)
    bmaj, bmin = hdr["BMAJ"], hdr["BMIN"]
    if bmaj <= 0 or bmin <= 0:
        raise RuntimeError(f"{restored_fits} has BMAJ={bmaj}, BMIN={bmin}; cannot convert")
    bpa = math.radians(hdr.get("BPA") or 0.0)
    cd = abs(hdr["CDELT1"]) * 3600.0  # arcsec/pixel

    # clean-beam kernel, peak 1, FITS PA convention (from north through
    # east; with CDELT1<0 east is -x, north is +y)
    ny, nx = m.shape
    pady, padx = ny // 2, nx // 2
    K = 2 * np.sqrt(2.0 * np.log(2.0))
    smaj, smin = bmaj * 3600.0 / (K * cd), bmin * 3600.0 / (K * cd)
    yy, xx = np.mgrid[0:2 * ny, 0:2 * nx]
    dx = (xx - nx) * 1.0
    dy = (yy - ny) * 1.0
    t = dx * (-math.sin(bpa)) + dy * math.cos(bpa)
    s = dx * math.cos(bpa) + dy * math.sin(bpa)
    ker = np.exp(-0.5 * ((t / smaj) ** 2 + (s / smin) ** 2))
    ker /= ker.sum()  # unit flux: conv sums to the model flux (Jy/px)

    mpad = np.pad(m, ((pady, ny - pady), (padx, nx - padx)))
    conv = fftconvolve(mpad, ker, mode="same")[pady:pady + ny, padx:padx + nx]

    psf_sum = float(psf.sum())
    factor_resid = float(psf.max()) / psf_sum
    a_px = hdr["CDELT1"] ** 2
    a_fit = math.pi * bmaj * bmin / (4.0 * math.log(2.0))
    a_eff = a_px * psf_sum / float(psf.max())
    out = conv + r * factor_resid
    hdr["BUNIT"] = "Jy/pixel"
    fits.writeto(out_fits, out, hdr, overwrite=True)
    return {"bmaj_deg": bmaj, "bmin_deg": bmin, "bpa_deg": hdr.get("BPA"),
            "beam_area_fit_deg2": a_fit, "beam_area_eff_deg2": a_eff,
            "beam_area_eff_over_fit": a_eff / a_fit,
            "psf": str(psf_fits), "psf_sum": psf_sum,
            "resid_beam_to_pix_factor": factor_resid,
            "model_flux_jy": float(m.sum()),
            "residual_flux_jy": float(r.sum() * factor_resid),
            "conversion": ("rebuilt term-by-term: model (*) clean beam + "
                           "residual/A_eff (dirty-PSF sum, flux-exact)")}


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
exportfits(imagename={prefix!r} + ".psf", fitsimage={psf_out!r},
           overwrite=True)
exportfits(imagename={prefix!r} + ".model", fitsimage={model_out!r},
           overwrite=True)
exportfits(imagename={prefix!r} + ".residual", fitsimage={residual_out!r},
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
    tag = f"_{args.tag}" if args.tag else ""
    work_dir = exp_dir / f".dirty_pass{tag}"
    cell, cell_source = resolve_cell(args, ms, mat)
    out_fits = exp_dir / f"{args.target}_dirty{tag}.fits"
    log_path = exp_dir / f"{args.target}_dirty{tag}.log"

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
    method_dir = run_dir(exp_dir, args.target, "casa", args.tag)
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
    psf_out = method_dir / f"{args.target}_casa_psf.fits"
    model_out = method_dir / f"{args.target}_casa-model.fits"
    residual_out = method_dir / f"{args.target}_casa-residual.fits"
    script = _CASA_CLEAN_SCRIPT.format(ms=str(ms), prefix=prefix, cell=f"{cell}arcsec",
                                       imsize=args.imsize, robust=args.robust,
                                       threshold=threshold_str, fits_out=str(out_fits),
                                       psf_out=str(psf_out), model_out=str(model_out),
                                       residual_out=str(residual_out))
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
    conv = jybeam_to_jypixel(out_fits, out_fits, model_fits=model_out,
                             residual_fits=residual_out, psf_fits=psf_out)
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
    method_dir = run_dir(exp_dir, args.target, "wsclean", args.tag)
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
    conv = jybeam_to_jypixel(out_fits, out_fits,
                             model_fits=Path(f"{prefix}-model.fits"),
                             residual_fits=Path(f"{prefix}-residual.fits"),
                             psf_fits=Path(f"{prefix}-psf.fits"))
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

    # Default: full-baseline (no-cutoff) mat at the common grid, super_res 1.0
    # (the uv_cutoff=2e6 mat is dropped). --nocut kept as an explicit alias for
    # the raw nocut mat at its own nominal scale (probe, declared).
    # --allch selects the all-channels mat (every channel of every SPW).
    mat_nocut = DATA_DIR / args.target / f"{args.target}_nocut.mat"
    mat_allch = DATA_DIR / args.target / f"{args.target}_allch.mat"
    mat_avg = DATA_DIR / args.target / f"{args.target}_avg.mat"
    if args.avg and args.allch:
        raise SystemExit("--avg and --allch are mutually exclusive")
    if args.avg:
        if not mat_avg.exists():
            raise SystemExit(f"--avg requested but not found: {mat_avg}")
        data_file = mat_avg
        mode_tag = "r2d2_avg"
        cell_note = ("channel-averaged mat: all channels of each SPW collapsed to 1 "
                     "noise-weighted visibility per row (2.31M vis, continuum mode; "
                     "uv at the mean frequency of the valid channels). Recovers the "
                     "SPWs whose ch0 is fully flagged (the v1/nocut mats had "
                     "effectively 10/24 SPWs). Loses intra-SPW uv coverage vs "
                     "--allch (declared).")
    elif args.allch:
        if not mat_allch.exists():
            raise SystemExit(f"--allch requested but not found: {mat_allch}")
        data_file = mat_allch
        mode_tag = "r2d2_allch"
        cell_note = ("all-channels mat: every channel of every SPW (24.86M vis), "
                     "frequency embedded per-channel in the uv coordinates (lambda). "
                     "nominal_pixelsize 0.0215 arcsec (max baseline at 247.8 GHz). "
                     "NOTE: the old 1-channel mat effectively held only 10/24 SPWs: "
                     "freqid=0 hits fully-flagged edge channels of the 16-ch SPWs.")
    elif args.nocut:
        if not mat_nocut.exists():
            raise SystemExit(f"--nocut requested but not found: {mat_nocut}")
        data_file = mat_nocut
        mode_tag = "r2d2_nocut"
        cell_note = ("no uv_cutoff: full baselines to 4.47e6 lambda, "
                     "nominal_pixelsize 0.0231 arcsec (probe off training scale). "
                     "The mat holds ONE channel per SPW "
                     "(freqid=0), both polarizations combined into one "
                     "Stokes-I visibility per row, unflagged rows only "
                     "(786684 vis) — and effectively only 10/24 SPWs, see --allch. "
                     "casa/wsclean mfs use every channel.")
    else:
        # Default: full-baseline (no-cutoff) mat at the common casa/wsclean
        # grid (0.047602 arcsec) with super_resolution 1.0: R2D2 derives
        # cell = nominal_pixelsize / super_resolution, so the mat is copied
        # with nominal_pixelsize set to the common cell (uv unchanged;
        # image-domain rescale, declared). The uv_cutoff=2e6 mat is dropped.
        if not mat_nocut.exists():
            raise SystemExit(f"default (no-cutoff) mat not found: {mat_nocut}")
        data_file = exp_dir / f"{args.target}_r2d2.mat"
        make_rescaled_mat(mat_nocut, data_file, COMMON_CELL)
        mode_tag = "r2d2"
        cell_note = ("no uv_cutoff: full baselines to 4.47e6 lambda, rescaled "
                     "copy at the common casa/wsclean grid (nominal_pixelsize "
                     "0.047602 arcsec, super_resolution 1.0; uv unchanged, "
                     "image-domain rescale). ONE channel per SPW (freqid=0), "
                     "both polarizations combined into one Stokes-I "
                     "visibility per row, unflagged rows only (786684 vis) — "
                     "effectively only 10/24 SPWs, see --allch/--avg. "
                     "casa/wsclean mfs use every channel.")
    variant = {"r2d2": None, "r2d2_allch": "allch", "r2d2_nocut": "nocut",
               "r2d2_avg": "avg"}[mode_tag]
    method_dir = run_dir(exp_dir, args.target, "r2d2", variant)

    r2d2_repo = BASE_DIR / "external" / "R2D2-RI"
    venv_python = BASE_DIR / ".envs" / ".r2d2" / "bin" / "python"
    ckpt = r2d2_repo / "ckpt" / "R2D2_small_ALMA3"
    # The yaml pins meas_op_on_gpu: True (un-overridable store_true CLI flag);
    # --cpu-op swaps in a yaml variant with the operator on CPU (24.9M-point
    # NUFFT OOMs on the 2GB GPU); the unet prior still runs on GPU.
    config = (BASE_DIR / "config" / "imaging" / "R2D2_small_cpuop.yaml" if args.cpu_op
              else BASE_DIR / "config" / "imaging" / "R2D2_small.yaml")
    cmd = [str(venv_python), "src/imager.py",
           "--config", str(config),
           "--output_path", str(method_dir),
           "--data_file", str(data_file),
           "--save_all_outputs",
           "--architecture", "unet",
           "--prune",
           "--sigma_res_tol", "1e-4",
           "--super_resolution", str(args.super_res),
           *( [] if args.cpu_op else ["--meas_op_on_gpu"] ),
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
    _mat = _loadmat(data_file)
    nominal_cell = float(_mat["nominal_pixelsize"].item())
    _nvis = int(_mat["u"].shape[0])
    # The branch notes carry HD163296's counts; report this mat's actual one.
    cell_note = cell_note.replace("2.31M vis", f"{_nvis/1e6:.2f}M vis")
    cell_note = cell_note.replace("24.86M vis", f"{_nvis/1e6:.2f}M vis")
    cell_note = cell_note.replace("786684 vis", f"{_nvis} vis")
    print("[r2d2] target:", args.target)
    print("[r2d2] data file:", data_file)
    print("[r2d2] mat:", cell_note)
    print("[r2d2] im_dim: %d x %d | super_resolution: %s -> cell %.6f arcsec"
          % (args.imsize, args.imsize, args.super_res, nominal_cell / args.super_res))
    print("[r2d2] weights: briggs %.1f | num_iter: 5 (k5) | ckpt: R2D2_small_ALMA3" % args.robust)
    if args.nocut or args.allch or args.avg:
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
        "mode": mode_tag,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "ms": str(ms),
        "data_file": str(data_file),
        "mat_note": cell_note,
        "tool": "R2D2-RI imager.py" + (" (probe off training scale, declared)" if (args.nocut or args.allch or args.avg) else ""),
        "cmd": cmd,
        "im_dim": args.imsize,
        "super_resolution": args.super_res,
        "cell_arcsec": cell_used,
        "weighting": f"briggs {args.robust}",
        "meas_op_device": "cpu" if args.cpu_op else "gpu",
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


def deepinv_cell_from_r2d2(exp_dir: Path, target: str, nocut_mat: Path,
                           variant: str | None = "nocut",
                           fallback_super_res: float = 0.4847) -> tuple[float, str]:
    """Cell matching the accepted r2d2 run (same grid for both regularized
    methods); falls back to mat_nominal / fallback_super_res."""
    from astropy.io import fits

    r2d2_fits = run_dir(exp_dir, target, "r2d2", variant) / f"{target}_r2d2.fits"
    if r2d2_fits.exists():
        cell = abs(float(fits.getheader(r2d2_fits)["CDELT2"])) * 3600.0
        return cell, f"accepted r2d2 grid ({r2d2_fits.name} CDELT2)"
    return (cell_from_mat(nocut_mat) / fallback_super_res,
            f"mat nominal_pixelsize / {fallback_super_res} (r2d2 super-res)")


def cmd_deepinv(args) -> None:
    ms, mat = target_paths(args.target)
    exp_dir = EXPERIMENTS_DIR / args.target
    if args.avg and args.allch:
        raise SystemExit("--avg and --allch are mutually exclusive")
    variant = "avg" if args.avg else ("allch" if args.allch else None)
    if args.tag:
        variant = f"{variant}_{args.tag}" if variant else args.tag
    method_dir = run_dir(exp_dir, args.target, "deepinverse", variant)
    mat_avg = DATA_DIR / args.target / f"{args.target}_avg.mat"
    nocut_mat = DATA_DIR / args.target / f"{args.target}_nocut.mat"
    if args.avg:
        if not mat_avg.exists():
            raise SystemExit(f"--avg requested but not found: {mat_avg}")
        src_mat = mat_avg
    else:
        src_mat = DATA_DIR / args.target / f"{args.target}_allch.mat" if args.allch else nocut_mat
    r2d2_variant = ("avg" if args.avg else ("allch" if args.allch else None))
    imsize = 64 if args.imsize == DEFAULT_IMSIZE else args.imsize
    cell, cell_source = deepinv_cell_from_r2d2(exp_dir, args.target, src_mat,
                                               variant=r2d2_variant,
                                               fallback_super_res=args.super_res)
    rescaled_mat = exp_dir / f"{args.target}_deepinv{'_avg' if args.avg else ('_allch' if args.allch else '')}.mat"
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
    print("[deepinv] source mat:", src_mat)
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
    make_rescaled_mat(src_mat, rescaled_mat, cell)

    # 3) run deepinverse (clears out_dir contents itself).
    method_dir.mkdir(parents=True, exist_ok=True)
    venv_python = BASE_DIR / ".envs" / ".deepinv" / "bin" / "python"
    env = dict(os.environ, PYTHONPATH=str(BASE_DIR))
    if args.cpu_op:
        # 2GB GPU: the 24.9M-point NUFFT OOMs on cuda; hide the GPU so
        # dinv.utils.get_device() falls back to cpu.
        env["CUDA_VISIBLE_DEVICES"] = ""
        print("[deepinv] device: CPU (CUDA hidden, --cpu-op; 24.9M-point NUFFT OOMs on the 2GB GPU)")
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

    common_note = ("Weighting: natural only (mat nW, sqrt of MS "
                   "natural weights); briggs=1 — declared deviation, casa/wsclean "
                   "use briggs robust=2 (wsclean 3.7 cannot persist imaging "
                   "weights on this multi-spw MS). Both polarizations combined "
                   "into one Stokes-I visibility per row, unflagged rows only.")
    if args.avg:
        mat_note = ("rescaled copy of the CHANNEL-AVERAGED mat: all channels of "
                    "each SPW collapsed to 1 noise-weighted visibility per row "
                    "(2.31M vis, continuum mode; uv at the mean frequency of the "
                    "valid channels); recovers the SPWs whose ch0 is fully flagged "
                    "(v1/nocut had effectively 10/24 SPWs) but loses intra-SPW uv "
                    "coverage vs the all-channels mat (declared). "
                    "nominal_pixelsize set to the accepted r2d2 cell; uv unchanged "
                    "(image-domain rescale, declared). w-term carried in the mat "
                    "but ignored by the 2D NUFFT (declared). " + common_note)
    elif args.allch:
        mat_note = ("rescaled copy of the ALL-CHANNELS mat: every channel of every "
                    "SPW (24.86M vis), frequency embedded per-channel in the uv "
                    "coordinates (lambda); nominal_pixelsize set to the accepted "
                    "r2d2 cell; uv unchanged (image-domain rescale, declared). "
                    "w-term carried in the mat but ignored by the 2D NUFFT "
                    "(declared). " + common_note)
    else:
        mat_note = ("rescaled copy of the no-cutoff mat: nominal_pixelsize set to "
                    "the accepted r2d2 cell; uv unchanged (image-domain rescale, "
                    "declared). " + common_note + " The mat holds ONE channel per "
                    "SPW (freqid=0) — and effectively only 10/24 SPWs, since ch 0 "
                    "of the 16-ch SPWs is fully flagged (786684 vis). casa/wsclean "
                    "mfs use every channel of every SPW.")

    from scipy.io import loadmat as _loadmat_di
    _nvis_di = int(_loadmat_di(src_mat)["u"].shape[0])
    # The branch notes carry HD163296's counts; report this mat's actual one.
    mat_note = mat_note.replace("2.31M vis", f"{_nvis_di/1e6:.2f}M vis")
    mat_note = mat_note.replace("24.86M vis", f"{_nvis_di/1e6:.2f}M vis")
    mat_note = mat_note.replace("786684 vis", f"{_nvis_di} vis")

    log = {
        "target": args.target,
        "mode": "deepinv_avg" if args.avg else ("deepinv_allch" if args.allch else "deepinv"),
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "ms": str(ms),
        "data_file": str(rescaled_mat),
        "mat_note": mat_note,
        "tool": "deepinverse.py (FISTA + WaveletPrior, l1_i50_v2 + fluxcal)",
        "cell_arcsec": cell,
        "cell_source": cell_source,
        "device": "cpu" if args.cpu_op else "gpu (dinv.utils.get_device())",
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


def _hires_beam_arcsec(target: str, robust: float) -> tuple[float, float, float] | None:
    """Synthesized beam (BMAJ, BMIN arcsec, BPA deg) from the high-res PSF dirty.

    casa/wsclean fit the beam on the recon grid (64x64), where the PSF is
    under-sampled and the fitted size comes out overestimated (up to ~2x on
    these elongated beams). The 512x512 @ cell/8 dirty carries a clean fit in
    its FITS header — that is the beam to draw. Returns None if the product
    is missing (caller keeps the header beam).
    """
    tag = "r0p5" if robust < 1.0 else "r2"
    p = EXPERIMENTS_DIR / target / f"{target}_dirty_hires_{tag}.fits"
    if not p.exists():
        return None
    from astropy.io import fits as _fits
    h = _fits.getheader(p)
    if "BMAJ" not in h or "BMIN" not in h or h["BMAJ"] <= 0 or h["BMIN"] <= 0:
        return None
    return (float(h["BMAJ"]) * 3600.0, float(h["BMIN"]) * 3600.0,
            float(h.get("BPA", 0.0)))


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


def _dirty_sigma_jybeam(target: str) -> tuple[float, str]:
    """Sigma (Jy/beam) of the target's full-FOV dirty image, as hand-measured.

    Primary source: the canonical casa run log — `threshold_jybeam` is the
    3-sigma absolute threshold whose note records that sigma was measured by
    hand on the full-FOV dirty image. Fallback: corner-patch estimate on
    <target>_dirty.fits. Returns (sigma_jybeam, source note).
    """
    from astropy.io import fits as _fits
    casa_dir = EXPERIMENTS_DIR / target / f"{target}_casa"
    for lp in sorted(casa_dir.glob(f"**/{target}_casa.log")):
        try:
            d = json.loads(lp.read_text())
        except (ValueError, OSError):
            continue
        th = d.get("threshold_jybeam")
        if th:
            return float(th) / 3.0, f"hand-measured via casa log (threshold/3)"
    p = EXPERIMENTS_DIR / target / f"{target}_dirty.fits"
    img = np.squeeze(_fits.getdata(p))  # casa exports (1, 1, ny, nx)
    s, _n = _corner_rms(img)  # Jy/beam (estimator is unit-linear)
    return s, f"corner-patch on {p.name} (no casa log)"


def _dirty_sigma_display(target: str, unit: str,
                         cell_arcsec: float) -> tuple[float, str]:
    """Hand-measured dirty sigma converted to display units, with source note.

    Jy/arcsec^2 divides by the beam area; mJy/px multiplies by the recon
    pixel area over the beam area (beam from the dirty FITS header, the image
    sigma was measured on). This is the common asinh stretch reference
    (--png --asinh, --lam-sweep): comparable across methods and immune to
    reconstructions whose background is exactly zero.
    """
    from astropy.io import fits as _fits
    sigma_beam, src = _dirty_sigma_jybeam(target)
    hdr = _fits.getheader(EXPERIMENTS_DIR / target / f"{target}_dirty.fits")
    beam_area_as2 = (np.pi * (hdr["BMAJ"] * 3600.0) * (hdr["BMIN"] * 3600.0)
                     / (4.0 * np.log(2.0)))
    if unit == "Jy/arcsec²":
        return sigma_beam / beam_area_as2, src
    return sigma_beam * 1e3 * cell_arcsec ** 2 / beam_area_as2, src  # mJy/px


def _linear_asinh_ticks(vmax: float, lw: float) -> list[float]:
    """Linearly spaced colorbar ticks (0, step, 2*step, ...) for an asinh bar.

    The step is the smallest 1-2-5 value that keeps the labels from
    overlapping: on an asinh bar equal linear steps compress toward vmax
    (dp/dv shrinks as ~1/(vmax*span) at the top), so the binding constraint
    is the last kept pair being >= 14% of the bar apart in the transformed
    coordinate, with the last tick at <= 93% of the bar (a label centered
    further right would stick out over the next panel's colorbar).
    """
    span = float(np.arcsinh(vmax / lw))

    def p_of(v: float) -> float:
        return float(np.arcsinh(v / lw)) / span

    steps = sorted(m * 10.0 ** k for k in range(-12, 13) for m in (1, 2, 5))
    step = next(s for s in steps if s >= vmax / 5.5)
    while True:
        ticks = [i * step for i in range(int(vmax / step + 1e-9) + 1)]
        kept = [v for v in ticks if p_of(v) <= 0.93]
        if (len(kept) <= 1 or p_of(kept[-1]) - p_of(kept[-2]) >= 0.14
                or step >= vmax):
            return kept or [0.0]
        step = next(s for s in steps if s > step)


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
    # Three generations: v1 = 1-channel mat (effectively 10/24 SPWs);
    # avg = channel-averaged mat (2.31M vis, all 24 SPWs); all_channels =
    # full-band mat (24.9M vis). PNGs are grouped per target:
    # results/archival/<target>/<variant>/.
    if args.test or args.tuned:
        # Winner-config panel (experiment 1 test set): casa/wsclean keep the
        # archival 3*sigma manual threshold with default cyclefactor/mgain;
        # BPSA uses the winner lambda (0.1, l0p1_i50_v2) from a dedicated
        # avg_test run; r2d2 reuses the canonical avg run (already the winner
        # k5 config). --tuned additionally uses dedicated casa/wsclean runs
        # with briggs robust 0.5 (hand-tuned for the archival data).
        if args.test and args.tuned:
            raise SystemExit("--png: --test and --tuned are mutually exclusive")
        if args.allch or args.avg or args.units != "mjypx" or args.tag:
            raise SystemExit("--png --test/--tuned takes no other variant flags")
        label = "tuned" if args.tuned else "test"
        out_dir = BASE_DIR / "results" / "archival" / args.target
        out_png = out_dir / f"{args.target}_panel_{label}{'_asinh' if args.asinh else ''}.png"
        unit = "mJy/px"
        deepinv_variant = "avg_test"
        if args.bpsa_lambda:
            deepinv_variant = f"avg_{args.bpsa_lambda}"
            bpsa_title = f"BPSA (λ={args.bpsa_lambda[3:]})"
        else:
            bpsa_title = "BPSA (λ=0.1)"
        r2d2_dir = run_dir(exp_dir, args.target, "r2d2", "avg")
        cw_variant = "tuned" if args.tuned else None
        if args.tuned and args.cw_variant == "canonical":
            cw_variant = None
    else:
        if args.allch and args.avg:
            raise SystemExit("--png: --avg and --allch are mutually exclusive")
        variant = ("all_channels" if args.allch
                   else ("avg" if args.avg else "v1"))
        out_dir = BASE_DIR / "results" / "archival" / args.target / variant
        if args.units == "jyas2":
            unit = "Jy/arcsec²"
            out_png = out_dir / f"{args.target}_panel_jyas2{'_asinh' if args.asinh else ''}.png"
        else:
            unit = "mJy/px"
            out_png = out_dir / f"{args.target}_panel{'_asinh' if args.asinh else ''}.png"
        deepinv_variant = "allch" if args.allch else ("avg" if args.avg else None)
        if args.tag:
            deepinv_variant = (f"{deepinv_variant}_{args.tag}"
                               if deepinv_variant else args.tag)
        r2d2_dir = run_dir(exp_dir, args.target, "r2d2",
                           "allch" if args.allch else ("avg" if args.avg else None))
        bpsa_title = (f"BPSA (λ={args.tag[3:]})" if args.tag
                      else f"BPSA (λ={'1e-3' if (args.avg or args.allch) else '1e-1'})")
        cw_variant = None
    deepinv_dir = run_dir(exp_dir, args.target, "deepinverse", deepinv_variant)

    # DSHARP published reference: crop to ~the reconstructions' FOV so the
    # comparison is fair. The reference FITS sits at
    # data/archival/<target>/<target>.fits (inside the target directory, next
    # to the MS and the mats). Its native unit is Jy/beam.
    dsharp_path = DATA_DIR / args.target / f"{args.target}.fits"
    _rcell = _panel_load(r2d2_dir / f"{args.target}_r2d2.fits")[1]
    _fov = 64 * _rcell  # reconstructions' FOV (64 x cell)
    dsharp, _, beam_ds = _panel_load(dsharp_path, crop_fov_arcsec=_fov * 1.02)
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
    # Its flux is the sum over the ~FOV crop (same window as the
    # reconstructions), so the comparison is fair.
    rms_ds, n_ds = _corner_rms(dsharp)
    dr_ds = float(dsharp.max()) / rms_ds
    flux_ds = float(dsharp_jypix.sum())
    sub_ds = f"DR={dr_ds:.0f}\nrms={rms_ds:.3g}"
    print("[png]   DSHARP: rms=%.4g %s (%d/4 corner patches), peak=%.4g, DR=%.4g, "
          "flux=%.4g Jy" % (rms_ds, unit, n_ds, dsharp.max(), dr_ds, flux_ds))
    casa_dir = run_dir(exp_dir, args.target, "casa", cw_variant)
    wsclean_dir = run_dir(exp_dir, args.target, "wsclean", cw_variant)
    panels = [("DSHARP", sub_ds, dsharp, beam_ds,
               getattr(args, "lw_dsharp", None))]
    casa_title = "$\\mathtt{tclean}$" + (" (robust 0.5)" if cw_variant == "tuned" else "")
    wsclean_title = "WSClean" + (" (robust 0.5)" if cw_variant == "tuned" else "")
    for title, path, beam_override, lw_ov in [
        (casa_title, casa_dir / f"{args.target}_casa.fits", None,
         getattr(args, "lw_tclean", None)),
        (wsclean_title, wsclean_dir / f"{args.target}_wsclean.fits", None,
         getattr(args, "lw_wsclean", None)),
        (bpsa_title, deepinv_dir / f"{args.target}_deepinverse.fits", "none",
         getattr(args, "lw_bpsa", None)),
        ("R2Disk2", r2d2_dir / f"{args.target}_r2d2.fits", "none",
         getattr(args, "lw_r2d2", None)),
    ]:
        img, cell, beam = _panel_load(path)
        if beam_override == "none":
            beam = None  # super-res grids: no clean beam to draw
        elif beam is not None:
            # casa/wsclean fit the beam on the 64x64 recon grid (under-sampled
            # PSF -> overestimated size): draw the beam measured on the
            # 512x512 @ cell/8 dirty instead (matching robust).
            _rob = 0.5 if cw_variant == "tuned" else 2.0
            _hb = _hires_beam_arcsec(args.target, _rob)
            if _hb is not None:
                _bmaj, _bmin, _bpa = _hb
                print("[png]   %s: beam from hires PSF (robust %.1f): "
                      "%.4g x %.4g arcsec, PA %.1f deg (header 64x64 fit was "
                      "%.4g x %.4g)"
                      % (title.split(" (")[0], _rob, _bmaj, _bmin, _bpa,
                         beam[0] * cell, beam[1] * cell))
                beam = (_bmaj / cell, _bmin / cell, _bpa)
        if args.units == "jyas2":
            q = img / cell ** 2  # Jy/pixel -> Jy/arcsec^2
        else:
            q = img * 1e3  # Jy/pixel -> mJy/pixel
        flux = float(img.sum())  # Jy/pixel sum = total flux (unit-independent)
        rms, n_patches = _corner_rms(q)
        dr = float(q.max()) / rms
        t_img = _method_imaging_time(path)
        t_txt = f"t={_fmt_duration(t_img)} | " if t_img is not None else ""
        sub = f"{t_txt}DR={dr:.0f}\nrms={rms:.3g}"
        print("[png]   %s: rms=%.4g %s (%d/4 corner patches), peak=%.4g, "
              "DR=%.4g, flux=%.4g Jy, imaging t=%s"
              % (title, rms, unit, n_patches, q.max(), dr, flux,
                 _fmt_duration(t_img) if t_img is not None else "n/a"))
        panels.append((title, sub, q, beam, lw_ov))

    print("[png] target:", args.target)
    for title, sub, img, beam, _lw in panels:
        print("[png]   %s: %s, beam=%s" % (title.split(" (")[0], img.shape,
                                           "yes" if beam else "no"))
    print("[png] output:", out_png)
    if args.dry_run:
        print("[png] dry-run: not executing")
        return

    # Shared log-count y limits across all panels (mold: across the row).
    peak_ct = 1
    for _, _, img, _, _ in panels:
        peak_ct = max(peak_ct, hist_peak(img, float(img.min()), float(img.max())))
    hist_ylim = (0.8, peak_ct * 1.2)

    # asinh stretch (only with --asinh): common linear_width for the
    # reconstruction panels (ASINH_LW_FAC x the target's full-FOV dirty sigma,
    # display units); the DSHARP reference (col 0) gets its own lower width
    # (ASINH_LW_FAC x its own corner-patch rms — its background is orders of
    # magnitude quieter, so the common width leaves its faint ring at the
    # colormap floor). vmax stays per-panel (each image's own max); the dirty
    # reference makes the stretch immune to zero backgrounds.
    lw_asinh = None
    lw_dsharp = None
    if args.asinh:
        fac = (ASINH_LW_FAC if getattr(args, "asinh_lw_factor", None) is None
               else args.asinh_lw_factor)
        _sig, _src = _dirty_sigma_display(args.target, unit, _rcell)
        lw_asinh = max(fac * _sig, 1e-12)
        # DSHARP keeps its own width at the fixed module constant: the CLI
        # factor tunes the reconstructions' stretch, not the reference's.
        lw_dsharp = max(ASINH_LW_FAC * rms_ds, 1e-12)
        print("[png] asinh linear_width = %.4g x sigma_dirty = %.4g %s (%s); "
              "DSHARP own = %.4g (%g x its corner rms, fixed ASINH_LW_FAC)"
              % (fac, lw_asinh, unit, _src, lw_dsharp, ASINH_LW_FAC))

    # 3x2 layout: top row DSHARP/tclean/WSClean, bottom row BPSA/R2Disk2
    # centered. Six columns; each panel spans a column pair, the bottom pair
    # offset by one column (slots 1-2 and 3-4 of the 6). Below each row the
    # colorbar + histogram hangs ~0.6in out of the cell; the bottom row's
    # 3-line title hangs ~0.9in above its cell (both budgeted in the gaps).
    fig = plt.figure(figsize=(13.33, 14.4), dpi=150)
    outer = fig.add_gridspec(2, 6, left=0.088, right=0.992, top=0.920,
                             bottom=0.075, wspace=0.005, hspace=0.423)
    slots = [(0, 0), (0, 2), (0, 4), (1, 1), (1, 3)]
    imshow_kw = dict(origin="lower", interpolation="nearest", aspect="equal")
    hist_pairs, top_panels = [], []

    for col, ((row, c0), (title, sub, img, beam, lw_ov)) in enumerate(
            zip(slots, panels)):
        # row0 height == cell width: the square image fills its cell
        # horizontally, so the visible gap between panels is just wspace
        cell = outer[row, c0:c0 + 2].subgridspec(2, 1,
                                                 height_ratios=[3.85, 1.0],
                                                 hspace=0.093)
        ax_img = fig.add_subplot(cell[0])
        v0, v1 = float(img.min()), float(img.max())
        lw_panel = None
        if args.asinh:
            from matplotlib.colors import AsinhNorm
            # Per-method --lw-<method> override wins; else DSHARP gets its
            # own fixed width, the recons share --asinh-lw-factor x sigma.
            lw_panel = (lw_ov if lw_ov is not None
                        else (lw_dsharp if col == 0 else lw_asinh))
            im = ax_img.imshow(img, norm=AsinhNorm(linear_width=lw_panel,
                                                   vmin=0.0, vmax=v1),
                               cmap="inferno", **imshow_kw)
        else:
            im = ax_img.imshow(img, vmin=v0, vmax=v1, cmap="inferno",
                               **imshow_kw)
        ax_img.set_title(title if sub is None else f"{title}\n{sub}",
                         fontsize=19, pad=12)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_anchor("N")
        if beam is not None:
            bmaj_px, bmin_px, bpa = beam
            for ec, lw in (("black", 2.4), ("white", 1.2)):
                # matplotlib: major axis (height) points to angle+90; FITS BPA
                # (from north through east, east=left for CDELT1<0) lands at
                # bpa-90 on screen, so angle=bpa reproduces the beam.
                ax_img.add_patch(Ellipse(
                    (0.11 * img.shape[1], 0.11 * img.shape[0]),
                    width=bmin_px, height=bmaj_px, angle=bpa,
                    fill=False, ec=ec, lw=lw, zorder=5))
        ax_hist = fig.add_subplot(cell[1])
        style_hist(ax_hist, img, np.linspace(v0, v1, HIST_BINS + 1),
                   ylim=hist_ylim, ylabels=(col == 0))
        ax_hist.tick_params(labelsize=16, colors="0.25")
        ax_hist.set_xlabel(f"Valor ({unit})", fontsize=16, color="0.25")
        if col == 0:
            ax_hist.set_ylabel("count (log)", fontsize=16, color="0.25",
                               labelpad=2)
        hist_pairs.append((ax_img, ax_hist))
        top_panels.append((ax_img, im, v1, lw_panel))

    # Fit each histogram below its colorbar, leaving room for the colorbar
    # tick labels (0.03 fig frac ~ 16pt text + pad).
    fig.canvas.draw()
    offset = CB_PAD + CB_H + 0.03
    for ax_img, ax_hist in hist_pairs:
        p_img = ax_img.get_position()
        p_hist = ax_hist.get_position()
        ax_hist.set_position(
            [p_img.x0, p_img.y0 - offset - p_hist.height,
             p_img.width, p_hist.height])

    for ax, im, vmax, plw in top_panels:
        cb = add_colorbar(fig, im, ax)
        cb.ax.tick_params(labelsize=16)
        if args.asinh:
            # Labels linearly spaced (0, step, 2*step, ...): on the asinh bar
            # they compress toward vmax, so the step grows (1-2-5) until the
            # top-end labels stop overlapping. The auto asinh minor ticks no
            # longer line up with the labels: switch them off.
            ticks = _linear_asinh_ticks(vmax, plw)
            cb.set_ticks(ticks)
            labels = cb.ax.set_xticklabels([f"{v:g}" for v in ticks])
            from matplotlib import ticker as _mticker
            cb.ax.xaxis.set_minor_locator(_mticker.NullLocator())
        # Edge labels grow inward instead of sticking out: the first would
        # sit on the first panel's hist y-tick labels, the last on the next
        # panel's colorbar.
        fig.canvas.draw()
        labels = cb.ax.get_xticklabels()
        if len(labels) >= 2:
            # Keep labels centered on their ticks but nudge the edge ones
            # inward by half the text width (in axes fraction) so they don't
            # stick out onto the neighboring panel.
            bar_w = cb.ax.get_window_extent().width
            for lbl, sign in ((labels[0], +1), (labels[-1], -1)):
                w = lbl.get_window_extent().width
                x, y = lbl.get_position()
                lbl.set_x(x + sign * 0.5 * w / bar_w)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    # 3 px white border on the lateral edges
    from PIL import Image
    _img = Image.open(out_png)
    _canvas = Image.new("RGB", (_img.width + 6, _img.height), "white")
    _canvas.paste(_img, (3, 0))
    _canvas.save(out_png)
    print("[png] done:", out_png)


def cmd_tidy(args) -> None:
    """Render the minimal abstract comparison figure (no reconstruction).

    DSHARP reference + casa + wsclean (canonical robust 2 runs) + BPSA
    (avg_lam3e-4) + R2Disk2 (avg), all converted to surface brightness in
    mJy/arcsec^2 and shown on a single common asinh scale. Five panels
    (3 over 2), one colorbar, no histograms/DR/rms, imaging time under
    each reconstruction title, beams on
    DSHARP/tclean/WSClean only, common 0.5 arcsec scale bar per panel.
    See the module docstring for the unit conventions. Saved to
    results/archival/<target>/<variant>/tidy/<target>_continuum_comparison.png
    (300 dpi) + .pdf.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm
    from matplotlib.patches import Ellipse

    from astropy.io import fits

    exp_dir = EXPERIMENTS_DIR / args.target
    out_dir = (BASE_DIR / "results" / "archival" / args.target
               / ("all_channels" if args.allch else ("avg" if args.avg else "v1"))
               / "tidy")
    out_png = out_dir / f"{args.target}_comparison.png"
    out_pdf = out_dir / f"{args.target}_comparison.pdf"

    # --- product selection: canonical casa/wsclean runs (robust 2) ----------
    casa_dir = run_dir(exp_dir, args.target, "casa", None)
    wsclean_dir = run_dir(exp_dir, args.target, "wsclean", None)
    bpsa_variant = args.bpsa_lambda or "lam3e-4"
    # the deepinv run dir carries two tags (variant + lambda); build it directly
    _dv = ("allch" if args.allch else ("avg" if args.avg else ""))
    deepinv_dir = (exp_dir / f"{args.target}_deepinverse"
                   / f"{args.target}_deepinverse_{_dv + '_' if _dv else ''}{bpsa_variant}")
    r2d2_dir = run_dir(exp_dir, args.target, "r2d2",
                       "allch" if args.allch else ("avg" if args.avg else None))

    dsharp_path = DATA_DIR / args.target / f"{args.target}.fits"
    paths = {"DSHARP reference": dsharp_path,
             "tclean": casa_dir / f"{args.target}_casa.fits",
             "WSClean": wsclean_dir / f"{args.target}_wsclean.fits",
             "BPSA": deepinv_dir / f"{args.target}_deepinverse.fits",
             "R2Disk2": r2d2_dir / f"{args.target}_r2d2.fits"}
    for label, p in paths.items():
        if not p.exists():
            raise SystemExit(f"--tidy: missing product for {label}: {p}")

    # --- unit verification + conversion to mJy/arcsec^2 --------------------
    # Every conversion is taken from the FITS header actually read: BUNIT
    # must state the native unit, CDELT the pixel scale, BMAJ/BMIN the
    # beam where one applies. Nothing is inferred from the values.
    BEAM_FAC = math.pi / (4.0 * math.log(2.0))  # 1.133: Gaussian FWHM -> area
    imgs, cells, beams_as, info = {}, {}, {}, {}
    for label, p in paths.items():
        h = fits.getheader(p)
        bunit = (h.get("BUNIT") or "").strip()
        d = np.squeeze(fits.getdata(p)).astype(np.float64)
        cell = abs(float(h["CDELT1"])) * 3600.0  # arcsec/pixel, own WCS
        if "DSHARP" in label:
            if "JY/BEAM" not in bunit.upper():
                raise SystemExit(
                    f"--tidy: {p.name} BUNIT='{bunit}', expected Jy/beam; "
                    "stopping rather than guessing the unit")
            for k in ("BMAJ", "BMIN"):
                if k not in h or float(h[k]) <= 0:
                    raise SystemExit(f"--tidy: {p.name} lacks a valid {k}; "
                                     "cannot convert Jy/beam to surface brightness")
            bmaj_as, bmin_as = float(h["BMAJ"]) * 3600.0, float(h["BMIN"]) * 3600.0
            bpa = float(h.get("BPA", 0.0))
            omega = BEAM_FAC * bmaj_as * bmin_as  # arcsec^2
            q = d * 1e3 / omega
            conv = f"Jy/beam -> mJy/arcsec^2: x1000 / (1.133 x {bmaj_as:.4f} x {bmin_as:.4f}) = x{1e3 / omega:.4g}"
            beam = (bmaj_as, bmin_as, bpa)
        else:
            if bunit.upper() != "JY/PIXEL":
                raise SystemExit(
                    f"--tidy: {p.name} BUNIT='{bunit}', expected Jy/pixel; "
                    "stopping rather than guessing the unit")
            q = d * 1e3 / cell ** 2
            conv = f"Jy/pixel -> mJy/arcsec^2: x1000 / ({cell:.4f}^2) = x{1e3 / cell ** 2:.4g}"
            beam = None  # model images / restored images: beam set below
        imgs[label], cells[label] = q, cell
        beams_as[label] = beam
        info[label] = (p, d.shape, bunit, cell, conv,
                       float(q.min()), float(q.max()))

    # DSHARP: crop to the reconstructions' FOV for a fair comparison.
    _rcell = cells["R2Disk2"]
    _fov = 64 * _rcell
    n = int(round(_fov * 1.02 / cells["DSHARP reference"]))
    c = imgs["DSHARP reference"].shape[0] // 2
    imgs["DSHARP reference"] = imgs["DSHARP reference"][
        c - n // 2: c - n // 2 + n, c - n // 2: c - n // 2 + n]

    # casa/wsclean: the 64x64 header fit under-samples the PSF; draw the beam
    # measured on the 512x512 @ cell/8 dirty of the same robust instead.
    for label, tag in (("tclean", "casa"), ("WSClean", "wsclean")):
        hb = _hires_beam_arcsec(args.target, args.robust)
        if hb is None:
            print(f"[tidy] WARNING: no hires beam for robust {args.robust}; "
                  f"{label} beam comes from its own header")
            h = fits.getheader(paths[label])
            hb = (float(h["BMAJ"]) * 3600.0, float(h["BMIN"]) * 3600.0,
                  float(h.get("BPA", 0.0)))
        beams_as[label] = hb

    # --- verification dump --------------------------------------------------
    print("[tidy] unit verification (all from FITS headers, no empirical factors):")
    for label, (p, shape, bunit, cell, conv, qmin, qmax) in info.items():
        crop = "" if "DSHARP" not in label else f" -> crop {imgs[label].shape}"
        print(f"[tidy]   {label:<16} {p.name}")
        print(f"[tidy]     shape={shape}{crop}  BUNIT='{bunit}'  "
              f"pixel scale={cell:.6f} arcsec/px")
        print(f"[tidy]     conversion: {conv}")
        print(f"[tidy]     in mJy/arcsec^2: min={qmin:.3f}  max={qmax:.3f}")
    for label in ("tclean", "WSClean"):
        b = beams_as[label]
        print(f"[tidy]   {label} beam (hires PSF, robust {args.robust}): "
              f"{b[0]:.4f} x {b[1]:.4f} arcsec, PA {b[2]:.1f} deg")
    b = beams_as["DSHARP reference"]
    print(f"[tidy]   DSHARP beam (header): {b[0]:.4f} x {b[1]:.4f} arcsec, "
          f"PA {b[2]:.1f} deg")
    print("[tidy]   BPSA / R2Disk2: model images (not convolved with a clean "
          "beam) -> no beam drawn")

    # --- common asinh scale --------------------------------------------------
    vmax_data = max(float(q.max()) for q in imgs.values())
    vmin = 0.0
    # display vmax: data max rounded UP to the nearest 1-2-5 value so the
    # colorbar ends on a round number (the data never exceeds it)
    vmax = next(v for v in sorted(m * 10.0 ** k for k in range(-6, 12)
                                  for m in (1, 2, 2.5, 5)) if v >= vmax_data)
    # Per-method asinh linear_width: the --lw-<method> flags are in mJy/px of
    # the 64x64 grid (the old panel's display units); scale to mJy/arcsec^2
    # with that grid's pixel area (1/cell^2). DSHARP: 10 x its own corner rms
    # (same convention as --png --asinh), or --lw-dsharp in mJy/arcsec^2.
    lw = {}
    mjypx_to_as2 = 1.0 / _rcell ** 2
    for label, flag in (("tclean", args.lw_tclean), ("WSClean", args.lw_wsclean),
                        ("BPSA", args.lw_bpsa), ("R2Disk2", args.lw_r2d2)):
        lw[label] = (flag * mjypx_to_as2 if flag is not None else None)
    if args.lw_dsharp is not None:
        lw["DSHARP reference"] = args.lw_dsharp
    else:
        rms_ds, _n = _corner_rms(imgs["DSHARP reference"])
        lw["DSHARP reference"] = max(ASINH_LW_FAC * rms_ds, 1e-12)
    _fac = (ASINH_LW_FAC if args.asinh_lw_factor is None else args.asinh_lw_factor)
    _sig, _src = _dirty_sigma_display(args.target, "Jy/arcsec²", _rcell)
    for label in lw:
        if lw[label] is None:
            lw[label] = max(_fac * _sig * 1e3, 1e-12)  # sigma_dirty -> mJy/as^2
    print("[tidy] common scale: vmin=0, vmax=%.4g mJy/arcsec^2 "
          "(data max %.4g rounded up to 1-2-5)" % (vmax, vmax_data))
    for label in ("DSHARP reference", "tclean", "WSClean", "BPSA", "R2Disk2"):
        print(f"[tidy]   asinh linear_width[{label}] = {lw[label]:.4g} mJy/arcsec^2")
    print("[tidy]   (no --lw given for a panel: %.4g = %.4g x sigma_dirty "
          "Jy/arcsec^2, %s)" % (_fac * _sig * 1e3, _fac, _src))
    print("[tidy] NOTE: single colorbar, drawn with the DSHARP panel's width: "
          "the per-method widths differ slightly, so the bar is exact for the "
          "reference panel and representative for the rest.")

    if args.dry_run:
        print("[tidy] dry-run: not executing")
        return

    # --- figure: 3 over 2, single colorbar ----------------------------------
    fig = plt.figure(figsize=(10.0, 7.4), facecolor="white")
    # axes placed manually: square panels, small fixed horizontal gap (a
    # gridspec leaves each image centered inside a wider cell, which pads the
    # visible inter-panel gap). fig fraction != inches: the figure is not
    # square, so width and height fractions differ for a square panel.
    # No general title (academic figure for the thesis abstract).
    titles = {"DSHARP reference": "DSHARP reference", "tclean": "tclean",
              "WSClean": "WSClean", "BPSA": "BPSA", "R2Disk2": "R2Disk2"}
    row_labels = [("DSHARP reference", "tclean", "WSClean"),
                  ("BPSA", "R2Disk2")]
    panel_h = 0.30                     # square side, figure fraction (height)
    panel_w = panel_h * 7.4 / 10.0     # same side in inches (fig is 10 x 7.4)
    gap_x = 0.012                      # horizontal gap between panels
    gap_y = 0.088                      # vertical gap between rows (fits the
                                       # two-line reconstruction titles)
    y_top0 = 0.925                     # top of the upper row (no suptitle)

    def _row_x0(n: int) -> float:
        return (1.0 - (n * panel_w + (n - 1) * gap_x)) / 2.0

    positions = {}
    for r, labels_r in enumerate(row_labels):
        x0 = _row_x0(len(labels_r))
        y_top = y_top0 - r * (panel_h + gap_y)
        for i, label in enumerate(labels_r):
            positions[label] = [x0 + i * (panel_w + gap_x), y_top - panel_h,
                                panel_w, panel_h]
    imshow_kw = dict(origin="lower", interpolation="nearest", aspect="equal")
    ims, axes = {}, {}
    for label, pos in positions.items():
        ax = fig.add_axes(pos)
        q = imgs[label]
        im = ax.imshow(q, norm=AsinhNorm(linear_width=lw[label], vmin=vmin,
                                         vmax=vmax),
                       cmap="inferno", **imshow_kw)
        title = titles[label]
        if label != "DSHARP reference":
            t_img = _method_imaging_time(paths[label])
            if t_img is not None:
                title = f"{title}\nt = {_fmt_duration(t_img)}"
        ax.set_title(title, fontsize=12.5, pad=7)
        ax.set_xticks([])
        ax.set_yticks([])
        b = beams_as[label]
        if b is not None:
            bmaj_px = b[0] / cells[label]
            bmin_px = b[1] / cells[label]
            for ec, lw_e in (("black", 1.6), ("white", 0.8)):
                ax.add_patch(Ellipse(
                    (0.11 * q.shape[1], 0.11 * q.shape[0]),
                    width=bmin_px, height=bmaj_px, angle=b[2],
                    fill=False, ec=ec, lw=lw_e, zorder=5))
        # common angular scale bar: 0.5 arcsec, lower right
        bar_px = 0.5 / cells[label]
        x1 = 0.93 * q.shape[1]
        y0 = 0.07 * q.shape[0]
        ax.plot([x1 - bar_px, x1], [y0, y0], color="white", lw=1.4,
                solid_capstyle="butt", zorder=5)
        ax.plot([x1 - bar_px, x1], [y0, y0], color="black", lw=0.6,
                solid_capstyle="butt", zorder=6)
        ax.text(x1 - bar_px / 2, y0 + 0.035 * q.shape[0], '0.5$^{\\prime\\prime}$',
                color="white", fontsize=8.5, ha="center", zorder=6,
                path_effects=None)
        ims[label] = im
        axes[label] = ax

    # single horizontal colorbar, framed on the bottom-row panels: from the
    # left edge of BPSA to the right edge of R2Disk2, just below the row
    # (drawn with the DSHARP panel's asinh width; see the note above)
    fig.canvas.draw()
    p_bpsa = axes["BPSA"].get_position()
    p_r2d2 = axes["R2Disk2"].get_position()
    cb_h = 0.016
    cax = fig.add_axes([p_bpsa.x0, p_bpsa.y0 - 0.026,
                        p_r2d2.x1 - p_bpsa.x0, cb_h])
    cb = fig.colorbar(ims["DSHARP reference"], cax=cax, orientation="horizontal")
    cb.set_label("Surface brightness (mJy arcsec$^{-2}$)", fontsize=10.5)
    cb.ax.tick_params(labelsize=9)
    # 1-2-5 ladder of low/intermediate values + the exact vmax, filtered by
    # minimum spacing in the asinh-transformed coordinate so the labels never
    # collide on the bar (the bar spans ~1800 px: ~0.065 span ~= 120 px).
    span = float(np.arcsinh(vmax / lw["DSHARP reference"]))

    def _p_of(v: float) -> float:
        return float(np.arcsinh(v / lw["DSHARP reference"])) / span

    ladder = sorted(m * 10.0 ** k for k in range(-6, 8) for m in (1, 2, 5)
                    if m * 10.0 ** k < vmax)
    ticks, last_p = [], None
    for v in [0.0] + ladder:
        p = _p_of(v)
        if last_p is None or p - last_p >= 0.065:
            ticks.append(v)
            last_p = p
    # keep the exact max; drop the previous tick if it would collide with it
    if _p_of(vmax) - last_p < 0.065:
        ticks.pop()
    ticks.append(vmax)
    cb.set_ticks(ticks)
    cb.ax.set_xticklabels([f"{v:.4g}" for v in ticks])
    from matplotlib import ticker as _mticker
    cb.ax.xaxis.set_minor_locator(_mticker.NullLocator())
    fig.canvas.draw()
    labels = cb.ax.get_xticklabels()
    if len(labels) >= 2:
        bar_w = cb.ax.get_window_extent().width
        for lbl, sign in ((labels[0], +1), (labels[-1], -1)):
            w = lbl.get_window_extent().width
            x, y = lbl.get_position()
            lbl.set_x(x + sign * 0.5 * w / bar_w)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, facecolor="white",
                bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_pdf, facecolor="white",
                bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("[tidy] done:")
    print("[tidy]  ", out_png)
    print("[tidy]  ", out_pdf)
    print("[tidy] caption note: BPSA and R2Disk2 panels are model images "
          "(not convolved with a clean beam); tclean/WSClean/DSHARP beams and "
          "the 0.5 arcsec scale bar are drawn per panel.")


def cmd_png_diff(args) -> None:
    """Render the 3-row BPSA allch-vs-avg comparison PNG (no reconstruction).

    Row 1: --allch BPSA per lambda; row 2: --avg BPSA per lambda (same
    lambdas); row 3: pixel difference allch - avg with the diff-image rms in
    the title. Layout is identical to cmd_lam_sweep's strip (own colorbar +
    log-count histogram per panel, same sizes). Products must exist already
    (run --deepinv [--allch|--avg] [--tag lam<X>]).
    Saved to results/archival/<target>/<target>_bpsa_allch_vs_avg.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exp_dir = EXPERIMENTS_DIR / args.target
    t = args.target
    out_png = (BASE_DIR / "results" / "archival" / t
               / f"{t}_bpsa_allch_vs_avg.png")
    # Lambda grid shared by the allch and avg generations. The canonical
    # dirs hold 1e-3 on both sides; every other lambda has its own variant.
    lambdas = [("1e-2", "allch_lam1e-2", "avg_lam1e-2"),
               ("3e-3", "allch_lam3e-3", "avg_lam3e-3"),
               ("1e-3", "allch", "avg"),
               ("3e-4", "allch_lam3e-4", "avg_lam3e-4")]
    cols = []
    for lab, va, vv in lambdas:
        fa = run_dir(exp_dir, t, "deepinverse", va) / f"{t}_deepinverse.fits"
        fv = run_dir(exp_dir, t, "deepinverse", vv) / f"{t}_deepinverse.fits"
        for f in (fa, fv):
            if not f.exists():
                raise SystemExit(f"[png-diff] missing product: {f}")
        a, ca, _ = _panel_load(fa)
        v, cv, _ = _panel_load(fv)
        if a.shape != v.shape:
            raise SystemExit(f"[png-diff] shape mismatch: {fa} vs {fv}")
        if abs(ca - cv) > 1e-6:
            raise SystemExit(f"[png-diff] cell mismatch: {ca} vs {cv} arcsec")
        a, v = a * 1e3, v * 1e3  # -> mJy/pixel
        diff = a - v
        rms = float(np.sqrt(np.mean(diff ** 2)))
        print("[png-diff] λ=%s: diff rms=%.4g mJy/px, max|diff|=%.4g"
              % (lab, rms, float(np.abs(diff).max())))
        cols.append((lab, a, v, diff, rms))

    print("[png-diff] output:", out_png)
    if args.dry_run:
        print("[png-diff] dry-run: not executing")
        return

    unit = "mJy/px"
    unit_diff = "µJy/px"  # difference row: ~100x smaller values, friendlier unit
    # Row content, lam-sweep title style (DR/rms) for the reconstructions;
    # the difference row carries its full-grid rms in µJy.
    rows = []  # each entry: (title, sub, img, cmap, unit)
    for row_name, idx in [("allch", 1), ("avg", 2)]:
        row = []
        for lab, a, v, diff, rms in cols:
            img = a if idx == 1 else v
            rms_c, _n = _corner_rms(img)
            dr = float(img.max()) / rms_c
            row.append((f"{row_name} λ={lab}",
                        f"DR={dr:.0f}\n"
                        f"rms={rms_c:.3g} {unit}", img, "inferno", unit))
        rows.append(row)
    rows.append([(f"allch−avg λ={lab}", f"rms={rms*1e3:.4g} {unit_diff}",
                  diff * 1e3, "RdBu_r", unit_diff)
                 for lab, a, v, diff, rms in cols])

    # Same mold as cmd_lam_sweep: per-column image + own colorbar + log-count
    # histogram, one shared hist y scale per row (each row is a sweep strip).
    # One deviation, per request: thinner colorbar than the mold's CB_H
    # (same full panel width as the mold's add_colorbar).
    from src.utils.studies.noisevar.plot_noisevar_comparisons import (
        HIST_BINS, CB_PAD, hist_peak, style_hist)

    CB_H_THIN = 0.006  # horizontal colorbar thickness (figure fraction)

    def thin_colorbar(fig, im, ax):
        p = ax.get_position()
        cax = fig.add_axes([p.x0, p.y0 - CB_PAD - CB_H_THIN,
                            p.x1 - p.x0, CB_H_THIN])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=10)
        from matplotlib.ticker import MaxNLocator
        cb.ax.xaxis.set_major_locator(MaxNLocator(5))
        return cb

    hist_ylims = []
    for row in rows:
        peak_ct = 1
        for _, _, img, _, _ in row:
            peak_ct = max(peak_ct, hist_peak(img, float(img.min()),
                                             float(img.max())))
        hist_ylims.append((0.8, peak_ct * 1.2))

    n = len(cols)
    fig = plt.figure(figsize=(3.1 * n, 6.0 * 3), dpi=150)
    outer = fig.add_gridspec(3, n, left=0.04, right=1 - 0.04, top=0.90,
                             bottom=0.04, hspace=0.65, wspace=0.02)
    imshow_kw = dict(origin="lower", interpolation="nearest", aspect="equal")
    hist_pairs, top_panels = [], []

    for r, row in enumerate(rows):
        for col, (title, sub, img, cmap, unit_c) in enumerate(row):
            cell = outer[r, col].subgridspec(2, 1, height_ratios=[3.4, 1.3],
                                             hspace=0.25)
            ax_img = fig.add_subplot(cell[0])
            if cmap == "RdBu_r":
                m = float(np.abs(img).max())
                v0, v1 = -m, m
            else:
                v0, v1 = float(img.min()), float(img.max())
            im = ax_img.imshow(img, vmin=v0, vmax=v1, cmap=cmap, **imshow_kw)
            ax_img.set_title(f"{title}\n{sub}", fontsize=12, pad=6)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            ax_img.set_anchor("N")
            ax_hist = fig.add_subplot(cell[1])
            style_hist(ax_hist, img, np.linspace(v0, v1, HIST_BINS + 1),
                       ylim=hist_ylims[r], ylabels=(col == 0))
            ax_hist.tick_params(labelsize=10.5)
            ax_hist.set_xlabel(f"Value ({unit_c})", fontsize=10.5, color="0.25")
            if col == 0:
                ax_hist.set_ylabel("count (log)", fontsize=10.5, color="0.25",
                                   labelpad=2)
            hist_pairs.append((ax_img, ax_hist))
            top_panels.append((ax_img, im))

    # Fit each histogram right below its colorbar (mold's align step). The
    # extra gap covers the colorbar's tick-label height below the thin bar.
    fig.canvas.draw()
    offset = CB_PAD + CB_H_THIN + 0.022
    for ax_img, ax_hist in hist_pairs:
        p_img = ax_img.get_position()
        p_hist = ax_hist.get_position()
        ax_hist.set_position(
            [p_img.x0, p_img.y0 - offset - p_hist.height,
             p_img.width, p_hist.height])

    for ax, im in top_panels:
        thin_colorbar(fig, im, ax)

    fig.suptitle(f"{t} — datos de archivo — BPSA: all-channels vs avg, λ a λ "
                 f"(l1_i50_v2 resto fijo) @ 64x64, {ca:.4g}\"/px\n"
                 "fila 1: all-channels mat (24.9M vis); fila 2: "
                 "channel-averaged mat (2.31M vis); fila 3: resta píxel a "
                 "píxel (celdas difieren < 1e-6\", despreciable, declarado)\n"
                 "rms por parches de esquina (recons, estimador del sweep) y "
                 "RMS de toda la grilla (restas, en µJy/px); DR = pico/rms",
                 fontsize=12.5, y=0.996)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    print("[png-diff] done:", out_png)


def cmd_lam_sweep(args) -> None:
    """Render the deepinv lambda-sweep strip from existing products (no recon).

    One panel per lambda variant (run dir <target>_deepinverse[_lam<tag>] inside
    the <target>_deepinverse container).
    Shared vmax, native Jy/pixel. Each
    title reports imaging time + DR, and the corner-patch rms (the user's
    estimator from src/utils/dirty_noise.py, same as --png).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exp_dir = EXPERIMENTS_DIR / args.target
    # Three generations: v1 = 1-channel mat variants; all_channels (--allch) =
    # full-band mat (24.9M vis); avg (--avg) = channel-averaged mat (2.31M
    # vis). Same lambda set is attempted; missing runs are skipped.
    allch = getattr(args, "allch", False)
    avg = getattr(args, "avg", False)
    if allch and avg:
        raise SystemExit("--lam-sweep: --avg and --allch are mutually exclusive")
    variant_dir = "all_channels" if allch else ("avg" if avg else "v1")
    out_png = (BASE_DIR / "results" / "archival" / args.target / variant_dir
               / f"{args.target}_deepinv_lambda_sweep.png")
    t = args.target

    def _v(v):  # sweep variant name inside the deepinverse container
        root = "allch" if allch else ("avg" if avg else None)
        if root is None:
            return v
        return f"{root}_{v}" if v else root

    di = lambda v: run_dir(exp_dir, t, "deepinverse", _v(v))  # noqa: E731
    # Entries are discovered dynamically: the canonical dir (avg/allch: λ=1e-3;
    # v1: λ=1e-1) plus every _lam<tag> variant dir that exists, sorted by λ
    # descending (tag parsed as float). New sweep points only need a
    # --deepinv run with --tag lam<value>; nothing here is hardcoded.
    root = "allch" if allch else ("avg" if avg else None)
    cont = run_dir(exp_dir, t, "deepinverse").parent  # method-level container
    canon_value = 1e-3 if root else 1e-1
    found = [(canon_value, "λ=1e-3" if root else "λ=1e-1", di(None))]
    pat = (rf"{re.escape(t)}_deepinverse_{root}_lam(.+)$" if root
           else rf"{re.escape(t)}_deepinverse_lam(.+)$")
    if cont.is_dir():
        for d in cont.iterdir():
            m = re.match(pat, d.name)
            if not (m and d.is_dir()):
                continue
            try:
                val = float(m.group(1))
            except ValueError:
                continue
            found.append((val, f"λ={m.group(1)}", d))
    found.sort(key=lambda e: e[0], reverse=True)
    entries = [(label, d, f"{t}_deepinverse.fits") for _, label, d in found]

    panels = []
    cell_px = COMMON_CELL
    for label, method_dir, fits_name in entries:
        path = method_dir / fits_name
        if not path.exists():
            print("[lam-sweep] WARNING: missing %s, skipping" % path)
            continue
        img, cell_px, _ = _panel_load(path)
        img_mjy = img * 1e3  # Jy/pixel -> mJy/pixel
        rms, n_patches = _corner_rms(img_mjy)
        dr = float(img_mjy.max()) / rms
        t_img = _method_imaging_time(path)
        print("[lam-sweep] %s: rms=%.4g mJy/px (%d/4 corner patches), DR=%.4g"
              % (label, rms, n_patches, dr))
        panels.append((label, img, t_img))

    print("[lam-sweep] output:", out_png)
    if args.dry_run:
        print("[lam-sweep] dry-run: not executing")
        return
    if not panels:
        raise SystemExit("[lam-sweep] no variant products found")

    # Same mold as the comparison panel (cmd_png): per-column image +
    # own colorbar + log-count histogram, inferno, shared hist y scale.
    from src.utils.studies.noisevar.plot_noisevar_comparisons import (
        HIST_BINS, CB_H, CB_PAD, add_colorbar, hist_peak, style_hist)
    from matplotlib.patches import Ellipse  # noqa: F401 (mold parity)

    unit = "mJy/px"
    mold_panels = []  # (title, sub, img, beam)
    for label, img, t_img in panels:
        img_mjy = img * 1e3  # Jy/pixel -> mJy/pixel
        rms, _n = _corner_rms(img_mjy)
        dr = float(img_mjy.max()) / rms
        t_txt = f"t={_fmt_duration(t_img)} | " if t_img is not None else ""
        sub = f"{t_txt}DR={dr:.0f}\nrms={rms:.3g} {unit}"
        mold_panels.append((label, sub, img_mjy, None, rms))

    peak_ct = 1
    for _, _, img, _, _ in mold_panels:
        peak_ct = max(peak_ct, hist_peak(img, float(img.min()), float(img.max())))
    hist_ylim = (0.8, peak_ct * 1.2)

    n = len(mold_panels)
    # 3-column grid, as many rows as needed. Same mold proportions as the
    # comparison panel (cmd_png): per-panel image+hist height ratio 3.85:1,
    # same inner hspace, title/hist font sizes and hist offset, so a sweep
    # strip looks identical to a panel row.
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(13.33, 7.2 * nrows + (0.8 if nrows == 1 else 0)),
                     dpi=150)
    outer = fig.add_gridspec(nrows, ncols, left=0.06, right=0.995,
                             top=0.86 if nrows == 1 else 0.95, bottom=0.05,
                             wspace=-0.15, hspace=0.5)
    imshow_kw = dict(origin="lower", interpolation="nearest", aspect="equal")
    hist_pairs, top_panels = [], []

    # asinh stretch: common linear_width for every panel, factor x sigma of the
    # target's full-FOV dirty (display units) — comparable across lambdas and
    # immune to sparse (exactly-zero) backgrounds.
    sfac = (ASINH_LW_FAC_SWEEP if getattr(args, "asinh_lw_sweep", None) is None
            else args.asinh_lw_sweep)
    _sig, _src = _dirty_sigma_display(t, unit, cell_px)
    lw_asinh = max(sfac * _sig, 1e-12)
    print("[lam-sweep] asinh linear_width = %.4g x sigma_dirty = %.4g %s (%s)"
          % (sfac, lw_asinh, unit, _src))

    from matplotlib.colors import AsinhNorm
    for idx, (title, sub, img, beam, rms) in enumerate(mold_panels):
        row, col = divmod(idx, ncols)
        cell = outer[row, col].subgridspec(2, 1, height_ratios=[3.85, 1.0],
                                           hspace=0.093)
        ax_img = fig.add_subplot(cell[0])
        v0, v1 = float(img.min()), float(img.max())
        im = ax_img.imshow(img, norm=AsinhNorm(linear_width=lw_asinh, vmin=0.0,
                                               vmax=v1),
                           cmap="inferno", **imshow_kw)
        ax_img.set_title(f"{title}\n{sub}", fontsize=19, pad=12)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_anchor("N")
        ax_hist = fig.add_subplot(cell[1])
        style_hist(ax_hist, img, np.linspace(v0, v1, HIST_BINS + 1),
                   ylim=hist_ylim, ylabels=(col == 0))
        ax_hist.tick_params(labelsize=16, colors="0.25")
        ax_hist.set_xlabel(f"Valor ({unit})", fontsize=16, color="0.25")
        if col == 0:
            ax_hist.set_ylabel("count (log)", fontsize=16, color="0.25",
                               labelpad=2)
        hist_pairs.append((ax_img, ax_hist))
        top_panels.append((ax_img, im, v1))

    # Absolute colorbar thickness (inches, figure-fraction here): the module
    # CB_H is a fraction of total figure height, so it grows with the number
    # of rows while the panels stay the same size.
    CB_H_ABS = 0.18  # inches
    cb_h = CB_H_ABS / fig.get_size_inches()[1]
    fig.canvas.draw()
    offset = CB_PAD + cb_h + 0.015
    for ax_img, ax_hist in hist_pairs:
        p_img = ax_img.get_position()
        p_hist = ax_hist.get_position()
        ax_hist.set_position(
            [p_img.x0, p_img.y0 - offset - p_hist.height,
             p_img.width, p_hist.height])

    from matplotlib import ticker as _mticker
    for ax, im, vmax in top_panels:
        p = ax.get_position()
        cax = fig.add_axes([p.x0, p.y0 - CB_PAD - cb_h, p.width, cb_h])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=16)
        # Same treatment as cmd_png --asinh: linearly spaced labels (0, step,
        # 2*step, ...) and edge labels nudged inward by half their width so
        # they stay centered on the tick without sticking onto the neighbor.
        ticks = _linear_asinh_ticks(vmax, lw_asinh)
        cb.set_ticks(ticks)
        cb.ax.set_xticklabels([f"{v:g}" for v in ticks])
        cb.ax.xaxis.set_minor_locator(_mticker.NullLocator())
        fig.canvas.draw()
        labels = cb.ax.get_xticklabels()
        if len(labels) >= 2:
            bar_w = cb.ax.get_window_extent().width
            for lbl, sign in ((labels[0], +1), (labels[-1], -1)):
                w = lbl.get_window_extent().width
                x, y = lbl.get_position()
                lbl.set_x(x + sign * 0.5 * w / bar_w)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
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
    p.add_argument("--tidy", action="store_true",
                   help="render the minimal abstract figure (existing products, no "
                   "recon): DSHARP + casa + wsclean (canonical robust 2) + BPSA "
                   "(avg_lam3e-4) + R2Disk2 (avg), all in mJy/arcsec^2 on a single "
                   "common asinh scale, one colorbar, no histograms/metrics; saved "
                   "to results/archival/<target>/<variant>/tidy/<target>_continuum_comparison.png+.pdf")
    p.add_argument("--png", action="store_true", help="render only the comparison panel PNG from existing products (no reconstruction)")
    p.add_argument("--lambda-factor", type=float, default=None,
                   help="with --deepinv: override lambda_factor (lower = more data fidelity/detail)")
    p.add_argument("--tag", default=None,
                   help="with --deepinv/--casa/--wsclean/--dirty: output dir suffix "
                        "(<target>_<method>_<tag>) for parameter probes")
    p.add_argument("--units", choices=("mjypx", "jyas2"), default="mjypx",
                   help="with --png: display units - mJy/px per image pixel or "
                        "Jy/arcsec^2 surface brightness (Jy/pixel / cell^2); "
                        "the DSHARP reference converts via its beam area")
    p.add_argument("--asinh", action="store_true",
                   help="with --png: render the panels with an asinh color scale "
                        "(linear_width = ASINH_LW_FAC x sigma_dirty of the target's "
                        "full-FOV dirty, common to the reconstruction panels, same "
                        "as --lam-sweep; the DSHARP panel uses its own corner rms) "
                        "instead of linear; output <target>_panel_asinh.png")
    p.add_argument("--asinh-lw-factor", type=float, default=None,
                   help="with --png --asinh: override ASINH_LW_FAC "
                        "(linear_width = factor x sigma_dirty)")
    for _m in ("dsharp", "tclean", "wsclean", "bpsa", "r2d2"):
        p.add_argument(f"--lw-{_m}", type=float, default=None,
                       help=f"with --png --asinh: absolute asinh linear_width "
                            f"for the {_m} panel, in display units (overrides "
                            f"--asinh-lw-factor / the DSHARP fixed width)")
    p.add_argument("--asinh-lw-sweep", type=float, default=None,
                   help="with --lam-sweep: override ASINH_LW_FAC_SWEEP "
                        "(linear_width = factor x sigma_dirty)")
    p.add_argument("--lam-sweep", action="store_true",
                   help="render the deepinv lambda-variant strip (existing products, "
                        "no recon) with imaging time, corner-patch rms and DR per panel")
    p.add_argument("--test", action="store_true",
                   help="with --png: winner-config panel (experiment 1 test set) - "
                        "casa/wsclean canonical 3*sigma runs (briggs 2), BPSA from "
                        "the avg_test run (winner lambda 0.1), R2Disk2 canonical "
                        "avg run; output results/archival/<target>/<target>_panel_test.png")
    p.add_argument("--tuned", action="store_true",
                   help="with --png: like --test but casa/wsclean from dedicated "
                        "briggs robust 0.5 runs (--casa/--wsclean --tag tuned); "
                        "output results/archival/<target>/<target>_panel_tuned.png")
    p.add_argument("--cw-variant", choices=("tuned", "canonical"), default="tuned",
                   help="with --png --tuned: which casa/wsclean runs to show "
                        "(default tuned = briggs 0.5; canonical = robust 2 runs)")
    p.add_argument("--bpsa-lambda", default=None, metavar="lam<X>",
                   help="with --png --tuned/--test: BPSA variant tag instead of "
                        "the winner lambda 0.1, e.g. lam3e-4 for the avg_lam3e-4 run")
    p.add_argument("--png-diff", action="store_true",
                   help="render the 3-row BPSA allch-vs-v1 PNG (existing products, no "
                        "recon): allch / v1 / pixel difference with diff rms per panel")
    p.add_argument("--nocut", action="store_true", help="with --r2d2: use the full-baseline <target>_nocut.mat instead of the cutoff mat")
    p.add_argument("--allch", action="store_true", help="with --r2d2/--deepinv: use <target>_allch.mat (all channels of all SPWs, frequency embedded in uv; 24.9M vis)")
    p.add_argument("--avg", action="store_true", help="with --r2d2/--deepinv/--lam-sweep: use <target>_avg.mat (channel-averaged: 1 noise-weighted visibility per SPW per row; 2.31M vis)")
    p.add_argument("--cpu-op", action="store_true", help="with --r2d2: run the measurement operator on CPU (needed for the 24.9M-vis allch mat on a 2GB GPU)")
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

    if not args.dirty and not args.casa and not args.wsclean and not args.r2d2 and not args.deepinv and not args.png and not args.tidy and not args.lam_sweep and not args.png_diff:
        p.error("no mode selected (available: --dirty, --casa, --wsclean, --r2d2, --deepinv, --png, --tidy, --lam-sweep, --png-diff)")
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
    if args.tidy:
        cmd_tidy(args)
    if args.lam_sweep:
        cmd_lam_sweep(args)
    if args.png_diff:
        cmd_png_diff(args)


if __name__ == "__main__":
    main()
