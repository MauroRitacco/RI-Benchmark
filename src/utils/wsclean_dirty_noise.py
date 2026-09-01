"""Generate a WSClean dirty image and estimate its corner-patch noise.

This helper is used both by the sweep driver (run_sweep.py) and by the
standalone WSClean wrapper (run_wsclean.sh) so that both paths share the same
noise-estimation logic and the same per-target cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

# Add project root for imports when this file is invoked directly.
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.dirty_noise import (
    DEFAULT_PATCH_SIZE,
    DEFAULT_SOURCE_MARGIN,
    DEFAULT_STD_CONSISTENCY_FACTOR,
    NoiseResult,
    PatchStats,
    estimate_noise,
    log_noise_result,
)


def _weight_to_cli(weight: str | Iterable) -> list[str]:
    """Convert a weight spec into WSClean CLI tokens.

    Examples
    --------
    "briggs 2" -> ["-weight", "briggs", "2"]
    ["briggs", 2] -> ["-weight", "briggs", "2"]
    """
    if isinstance(weight, str):
        parts = weight.split()
    else:
        parts = [str(w) for w in weight]
    if not parts:
        return ["-weight", "briggs", "2"]
    return ["-weight"] + parts


def _build_dirty_pass_args(
    ms_path: str | Path,
    size: int,
    pixel_size: float,
    weight: str | Iterable,
    data_column: str,
    dirty_prefix: str | Path,
) -> list[str]:
    """Build the wsclean -niter 0 command for the dirty pass."""
    args = ["wsclean"]
    args += ["-size", str(size), str(size)]
    args += ["-scale", f"{pixel_size}asec"]
    args += _weight_to_cli(weight)
    # Use w-stacking with a single layer to match the 2D gridding used by CASA
    # standard gridder and DeepInverse's NUFFT.
    args += ["-gridder", "wstacking"]
    args += ["-wstack-nwlayers", "1"]
    args += ["-niter", "0"]
    args += ["-data-column", data_column]
    args += ["-pol", "I"]
    args += ["-j", "1"]
    args += ["-no-update-model-required"]
    args += ["-name", str(dirty_prefix)]
    args += [str(ms_path)]
    return args


def _read_cache(cache_path: Path) -> dict | None:
    """Return cached noise data if it exists and looks valid."""
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _cache_is_valid(cache: dict, ms_path: Path, size: int, pixel_size: float,
                    weight: str, data_column: str,
                    gridder: str = "wstacking", wstack_nwlayers: int = 1) -> bool:
    """Check whether the cache entry matches the requested dirty-pass parameters."""
    try:
        if not ms_path.exists():
            return False
        expected_mtime = ms_path.stat().st_mtime
        if cache.get("ms_mtime") != expected_mtime:
            return False
        if cache.get("size") != size:
            return False
        if not np.isclose(float(cache.get("pixel_size", 0)), float(pixel_size)):
            return False
        if cache.get("weight") != weight:
            return False
        if cache.get("data_column") != data_column:
            return False
        if cache.get("gridder") != gridder:
            return False
        if cache.get("wstack_nwlayers") != wstack_nwlayers:
            return False
        sigma = float(cache.get("sigma_noise", float("nan")))
        if not np.isfinite(sigma) or sigma <= 0:
            return False
        return True
    except Exception:
        return False


def _write_cache(cache_path: Path, result: NoiseResult, ms_path: Path,
                 size: int, pixel_size: float, weight: str, data_column: str,
                 gridder: str = "wstacking", wstack_nwlayers: int = 1) -> None:
    """Serialize the noise result together with the dirty-pass parameters."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "ms_path": str(ms_path),
        "ms_mtime": ms_path.stat().st_mtime if ms_path.exists() else None,
        "size": size,
        "pixel_size": float(pixel_size),
        "weight": weight,
        "data_column": data_column,
        "gridder": gridder,
        "wstack_nwlayers": wstack_nwlayers,
        "sigma_noise": result.sigma_noise,
        "multiplier": result.multiplier,
        "threshold": result.threshold,
        "valid_patches": [
            {"name": p.name, "mean": p.mean, "std": p.std, "rms": p.rms, "n": p.n}
            for p in result.valid_patches
        ],
        "discarded_patches": result.discarded_patches,
        "source_px": result.source_px,
        "has_point_source": result.has_point_source,
    }
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def get_wsclean_sigma_noise(
    ms_path: str | Path,
    size: int,
    pixel_size: float,
    weight: str | Iterable,
    data_column: str,
    dirty_prefix: str | Path,
    gt_path: str | Path | None = None,
    cache_path: str | Path | None = None,
    patch_size: int = DEFAULT_PATCH_SIZE,
    source_margin: int = DEFAULT_SOURCE_MARGIN,
    std_consistency_factor: float = DEFAULT_STD_CONSISTENCY_FACTOR,
    log_func: callable = logging.info,
    cwd: str | Path | None = None,
) -> NoiseResult:
    """Run a WSClean dirty pass (niter=0) and estimate noise from corner patches.

    Parameters
    ----------
    ms_path
        Path to the Measurement Set.
    size
        Image size in pixels (square).
    pixel_size
        Pixel size in arcseconds.
    weight
        WSClean weight spec, e.g. "briggs 2" or ["briggs", 2].
    data_column
        Visibility data column name, e.g. "DATA".
    dirty_prefix
        Prefix passed to WSClean ``-name`` for the dirty pass. The dirty image will
        be ``{dirty_prefix}-dirty.fits``.
    gt_path
        Ground-truth FITS containing HAS_EXT/EXT_X/EXT_Y for point-source filtering.
    cache_path
        Optional JSON cache path. If provided and the cache is valid, the dirty pass
        is skipped and the cached result is returned.
    log_func
        Logging callable used for diagnostic messages.

    Returns
    -------
    NoiseResult with sigma_noise, threshold (for multiplier=1.0) and patch details.

    Raises
    ------
    RuntimeError
        If the dirty pass fails, the dirty image is missing, or the noise estimate
        is invalid (NaN, zero, or no valid patches).
    """
    ms_path = Path(ms_path).resolve()
    dirty_prefix = Path(dirty_prefix).resolve()
    if gt_path is not None:
        gt_path = Path(gt_path).resolve()
    if cache_path is not None:
        cache_path = Path(cache_path).resolve()
    if cwd is not None:
        cwd = Path(cwd).resolve()
    weight_str = " ".join(_weight_to_cli(weight)[1:])

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache = _read_cache(cache_path)
        if cache is not None and _cache_is_valid(
            cache, ms_path, size, pixel_size, weight_str, data_column
        ):
            log_func("[noise] Using cached WSClean dirty noise: sigma=%.6e", cache["sigma_noise"])
            valid_patches = [
                PatchStats(name=p["name"], mean=p["mean"], std=p["std"], rms=p["rms"], n=p["n"])
                for p in cache.get("valid_patches", [])
            ]
            discarded_patches = list(cache.get("discarded_patches", []))
            source_px = cache.get("source_px")
            if source_px is not None:
                source_px = tuple(source_px)
            return NoiseResult(
                sigma_noise=cache["sigma_noise"],
                multiplier=1.0,
                threshold=cache["sigma_noise"],
                valid_patches=valid_patches,
                discarded_patches=discarded_patches,
                source_px=source_px,
                has_point_source=cache.get("has_point_source", False),
            )

    dirty_args = _build_dirty_pass_args(
        ms_path, size, pixel_size, weight, data_column, dirty_prefix
    )
    log_func("[noise] WSClean dirty pass: %s", " ".join(dirty_args))
    result = subprocess.run(dirty_args, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(
            f"WSClean dirty pass failed (exit {result.returncode}):\n{result.stderr}\n{result.stdout}"
        )

    dirty_image = Path(f"{dirty_prefix}-dirty.fits")
    if not dirty_image.exists():
        raise RuntimeError(f"WSClean dirty image not found: {dirty_image}")

    noise_result = estimate_noise(
        dirty_path=dirty_image,
        gt_path=gt_path,
        multiplier=1.0,
        patch_size=patch_size,
        source_margin=source_margin,
        std_consistency_factor=std_consistency_factor,
    )

    log_noise_result(noise_result, log_func=log_func)

    if cache_path is not None:
        _write_cache(
            cache_path, noise_result, ms_path, size, pixel_size, weight_str, data_column
        )
        log_func("[noise] Cached WSClean dirty noise to %s", cache_path)

    return noise_result


def main():
    parser = argparse.ArgumentParser(
        description="Run a WSClean dirty pass and estimate corner-patch noise."
    )
    parser.add_argument("--ms", required=True, help="Path to the Measurement Set")
    parser.add_argument("--size", type=int, default=64, help="Image size in pixels")
    parser.add_argument("--pixel-size", type=float, required=True, help="Pixel size in arcsec")
    parser.add_argument("--weight", default="briggs 2", help="WSClean weight spec")
    parser.add_argument("--data-column", default="DATA", help="Visibility data column")
    parser.add_argument("--dirty-prefix", required=True, help="Prefix for dirty pass -name")
    parser.add_argument("--gt-path", default=None, help="Ground-truth FITS for source filtering")
    parser.add_argument("--cache", default=None, help="JSON cache path")
    parser.add_argument("--multiplier", type=float, default=1.0,
                        help="Threshold multiplier to apply to sigma_noise")
    parser.add_argument("--log", default=None, help="Append diagnostic log to this file")
    args = parser.parse_args()

    log_func = logging.info
    if args.log:
        log_dir = Path(args.log).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(args.log, mode="a")
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger = logging.getLogger("wsclean_dirty_noise")
        logger.handlers = []
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
        log_func = logger.info

    result = get_wsclean_sigma_noise(
        ms_path=args.ms,
        size=args.size,
        pixel_size=args.pixel_size,
        weight=args.weight,
        data_column=args.data_column,
        dirty_prefix=args.dirty_prefix,
        gt_path=args.gt_path,
        cache_path=args.cache,
        log_func=log_func,
    )

    threshold = args.multiplier * result.sigma_noise
    print(f"sigma_noise={result.sigma_noise}")
    print(f"threshold={threshold}")


if __name__ == "__main__":
    main()
