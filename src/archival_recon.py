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

(future modes: casa / wsclean / deepinv / r2d2 reconstructions, panel)

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
import subprocess
import sys
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
from casatasks import tclean, exportfits
tclean(vis={ms!r}, imagename={prefix!r}, cell={cell!r}, imsize={imsize},
       specmode="mfs", deconvolver="hogbom", gridder="standard",
       weighting="briggs", robust={robust}, niter=1000000, gain=0.1,
       threshold={threshold!r}, cyclefactor=0.0, savemodel="none",
       stokes="I", datacolumn="data")
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
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not out_fits.exists():
        raise SystemExit(f"CASA clean failed (exit {res.returncode}):\n"
                         f"{res.stderr[-2000:]}\n{res.stdout[-2000:]}")
    conv = jybeam_to_jypixel(out_fits, out_fits)
    log = {
        "target": args.target,
        "mode": "casa",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "ms": str(ms),
        "tool": "casa tclean (hogbom)",
        "casa_cmd": f"tclean(vis, cell='{cell}arcsec', imsize={args.imsize}, "
                    f"specmode='mfs', deconvolver='hogbom', gridder='standard', "
                    f"weighting='briggs', robust={args.robust}, niter=1000000, "
                    f"gain=0.1, threshold='{threshold_str}', cyclefactor=0.0, "
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
    res = subprocess.run(cmd, capture_output=True, text=True)
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True, help="archival target name (data/archival/<target>)")
    p.add_argument("--dirty", action="store_true", help="generate only the full-resolution dirty image")
    p.add_argument("--casa", action="store_true", help="CASA tclean reconstruction (hogbom, 3*sigma absolute threshold)")
    p.add_argument("--wsclean", action="store_true", help="WSClean reconstruction (single-scale, 3*sigma absolute threshold)")
    p.add_argument("--threshold", type=float, default=None,
                   help="absolute threshold in Jy/beam, already multiplied by 3 "
                        "(e.g. rms 1.515e-3 -> pass 4.545e-3); required with --casa")
    p.add_argument("--imsize", type=int, default=DEFAULT_IMSIZE,
                   help="dirty image size in pixels (default: %(default)s)")
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

    if not args.dirty and not args.casa and not args.wsclean:
        p.error("no mode selected (available: --dirty, --casa, --wsclean)")
    if args.dirty:
        cmd_dirty(args)
    if args.casa:
        cmd_casa(args)
    if args.wsclean:
        cmd_wsclean(args)


if __name__ == "__main__":
    main()
