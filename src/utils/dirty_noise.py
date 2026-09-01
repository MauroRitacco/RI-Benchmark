"""Estimate noise in a dirty image from four corner patches.

The geometry and source filtering are driven by the synthetic-disk generator
(DiskGenerator/src/disk_generator.py).  For a 64x64 image the emission is
confined to the central ~32 px, so 9x9 corner patches are outside the disk and
star.  Patches that overlap a known random point source (read from the ground
truth FITS header) are discarded, and the remaining patches must pass a
purity test (rms ≈ std).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits


DEFAULT_PATCH_SIZE = 9
DEFAULT_SOURCE_MARGIN = 5
DEFAULT_STD_CONSISTENCY_FACTOR = 2.0


@dataclass
class Patch:
    """A square corner patch defined by array slices and a human-readable name."""

    name: str
    row_slice: slice
    col_slice: slice


@dataclass
class PatchStats:
    """Classical statistics for one patch."""

    name: str
    mean: float
    std: float
    rms: float
    n: int


@dataclass
class NoiseResult:
    """Result of the full noise-estimation pipeline."""

    sigma_noise: float
    multiplier: float
    threshold: float
    valid_patches: list[PatchStats]
    discarded_patches: list[dict]
    source_px: tuple[float, float] | None
    has_point_source: bool


def read_source_info(gt_path: str | Path) -> tuple[bool, float, float]:
    """Read whether a random point source exists and its pixel offsets.

    Returns
    -------
    has_source : bool
        Value of the HAS_EXT header (True/False).
    ext_x, ext_y : float
        Source offset from the image centre, in pixels, along axis 1 and 2.
        These are zero when no source is present.

    Raises
    ------
    RuntimeError
        If HAS_EXT is True but EXT_X/EXT_Y are missing from the header.
    """
    gt_path = Path(gt_path)
    hdr = fits.getheader(str(gt_path))

    has_ext_raw = hdr.get("HAS_EXT", None)
    if has_ext_raw is None:
        raise RuntimeError(
            f"{gt_path}: header keyword HAS_EXT is missing; cannot determine "
            "whether a point source is present."
        )

    has_source = str(has_ext_raw).strip().lower() in ("true", "1", "yes", "t")

    if not has_source:
        return False, 0.0, 0.0

    ext_x = hdr.get("EXT_X", None)
    ext_y = hdr.get("EXT_Y", None)
    if ext_x is None or ext_y is None:
        raise RuntimeError(
            f"{gt_path}: HAS_EXT is True but EXT_X/EXT_Y are missing; cannot "
            "locate the point source."
        )

    return True, float(ext_x), float(ext_y)


def source_pixel_in_dirty(
    gt_path: str | Path,
    dirty_path: str | Path,
    ext_x: float,
    ext_y: float,
) -> tuple[float, float]:
    """Map the source offset (EXT_X, EXT_Y) to pixel coordinates in the dirty image.

    EXT_X/EXT_Y are offsets from the image centre in pixels, as written by the
    disk generator.  The ground-truth and dirty images in this benchmark are on
    the same pixel grid, so the source lands at the same offset from each
    image's CRPIX.  We therefore transfer the pixel offset directly rather
    than going through sky coordinates, because the synthetic ground-truth FITS
    has CRVAL=(0,0) while CASA/WSClean dirty images inherit the observation
    phase centre as CRVAL; a WCS transfer would mix two different reference
    points and produce incorrect (often NaN) positions.
    """
    gt_path = Path(gt_path)
    dirty_path = Path(dirty_path)

    dirty_hdr = fits.getheader(str(dirty_path))
    crpix1 = float(dirty_hdr.get("CRPIX1", (dirty_hdr["NAXIS1"] + 1) / 2.0))
    crpix2 = float(dirty_hdr.get("CRPIX2", (dirty_hdr["NAXIS2"] + 1) / 2.0))

    # Source position in dirty-image FITS 1-based pixels.
    src_col_1based = crpix1 + ext_x
    src_row_1based = crpix2 + ext_y

    # Convert to 0-based numpy indices: (row, col).
    # FITS axis 1 ↔ array columns, axis 2 ↔ array rows.
    row = src_row_1based - 1.0
    col = src_col_1based - 1.0
    return row, col


def define_corner_patches(
    img_shape: tuple[int, ...] | np.ndarray,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> list[Patch]:
    """Return four 9x9 corner patches for a square image."""
    if len(img_shape) < 2:
        raise ValueError(f"img_shape must have at least 2 dimensions, got {img_shape}")
    ny, nx = int(img_shape[-2]), int(img_shape[-1])
    if patch_size > ny or patch_size > nx:
        raise ValueError(
            f"patch_size {patch_size} does not fit in image shape ({ny}, {nx})"
        )

    return [
        Patch("TL", slice(0, patch_size), slice(0, patch_size)),
        Patch("TR", slice(0, patch_size), slice(nx - patch_size, nx)),
        Patch("BL", slice(ny - patch_size, ny), slice(0, patch_size)),
        Patch("BR", slice(ny - patch_size, ny), slice(nx - patch_size, nx)),
    ]


def _patch_overlaps_source(
    patch: Patch,
    source_row: float,
    source_col: float,
    margin: int,
) -> bool:
    """Return True if the source (with margin) falls inside the patch."""
    rmin = patch.row_slice.start - margin
    rmax = patch.row_slice.stop + margin - 1
    cmin = patch.col_slice.start - margin
    cmax = patch.col_slice.stop + margin - 1
    return (rmin <= source_row <= rmax) and (cmin <= source_col <= cmax)


def filter_patches_by_source(
    patches: Iterable[Patch],
    source_row: float,
    source_col: float,
    margin: int = DEFAULT_SOURCE_MARGIN,
) -> tuple[list[Patch], list[dict]]:
    """Discard patches that overlap the point source (expanded by margin)."""
    valid: list[Patch] = []
    discarded: list[dict] = []
    for patch in patches:
        if _patch_overlaps_source(patch, source_row, source_col, margin):
            discarded.append(
                {
                    "patch": patch.name,
                    "reason": "overlaps point source",
                    "source_px": (source_row, source_col),
                }
            )
        else:
            valid.append(patch)
    return valid, discarded


def compute_patch_stats(
    data: np.ndarray, patch: Patch, detrend: bool = True
) -> PatchStats:
    """Classical mean, std and RMS of a patch.

    By default a 2D plane (z = a*x + b*y + c) is fit by least squares and the
    std/rms are computed from the residuals.  This removes a local sidelobe
    pedestal gradient from the noise estimate while keeping the original mean
    for diagnostic purposes.
    """
    values_2d = data[patch.row_slice, patch.col_slice]
    ny, nx = values_2d.shape
    mean = float(np.mean(values_2d))

    if detrend:
        y, x = np.mgrid[0:ny, 0:nx]
        A = np.c_[x.ravel(), y.ravel(), np.ones(nx * ny)]
        coef, *_ = np.linalg.lstsq(A, values_2d.ravel(), rcond=None)
        plane = (coef[0] * x + coef[1] * y + coef[2]).reshape(values_2d.shape)
        residuals = values_2d - plane
        std = float(np.std(residuals, ddof=0))
        rms = float(np.sqrt(np.mean(residuals**2)))
    else:
        residuals = values_2d
        std = float(np.std(values_2d, ddof=0))
        rms = float(np.sqrt(np.mean(values_2d**2)))

    return PatchStats(patch.name, mean, std, rms, int(values_2d.size))


def filter_patches_by_std_consistency(
    patch_stats: Iterable[PatchStats],
    factor: float = DEFAULT_STD_CONSISTENCY_FACTOR,
) -> tuple[list[PatchStats], list[dict]]:
    """Discard patches whose std deviates strongly from the median std.

    The corners of a dirty image can share a common sidelobe pedestal (captured
    in the patch mean) while still having the same noise std.  This filter keeps
    only patches whose std is within `factor` of the median std across all
    supplied patches.
    """
    patch_stats = list(patch_stats)
    if not patch_stats:
        return [], []

    stds = np.array([s.std for s in patch_stats])
    median_std = float(np.median(stds))
    if median_std == 0.0:
        # All patches are perfectly flat; treat none as valid.
        return [], [
            {"patch": s.name, "reason": "median std is zero", "std": s.std}
            for s in patch_stats
        ]

    valid: list[PatchStats] = []
    discarded: list[dict] = []
    for stats in patch_stats:
        if stats.std <= factor * median_std:
            valid.append(stats)
        else:
            discarded.append(
                {
                    "patch": stats.name,
                    "reason": f"std inconsistency ({stats.std:.4e} > {factor}*{median_std:.4e})",
                    "mean": stats.mean,
                    "std": stats.std,
                    "rms": stats.rms,
                }
            )
    return valid, discarded


def estimate_noise(
    dirty_path: str | Path,
    gt_path: str | Path | None = None,
    multiplier: float = 1.0,
    patch_size: int = DEFAULT_PATCH_SIZE,
    source_margin: int = DEFAULT_SOURCE_MARGIN,
    std_consistency_factor: float = DEFAULT_STD_CONSISTENCY_FACTOR,
    detrend: bool = True,
) -> NoiseResult:
    """Estimate sigma_noise from corner patches of a dirty image.

    Parameters
    ----------
    dirty_path
        Path to the dirty FITS image (in native units, e.g. Jy/beam).
    gt_path
        Path to the ground-truth FITS containing HAS_EXT/EXT_X/EXT_Y.  If None,
        no point-source filtering is performed.
    multiplier
        Factor by which sigma_noise is multiplied to build the threshold.
    std_consistency_factor
        Patches whose std is larger than this factor times the median std are
        discarded as contaminated.
    detrend
        If True, fit and remove a 2D plane from each patch before computing std.

    Returns
    -------
    NoiseResult with sigma_noise, threshold and per-patch details.
    """
    dirty_path = Path(dirty_path)
    data = fits.getdata(str(dirty_path)).squeeze().astype(np.float64)

    patches = define_corner_patches(data.shape, patch_size=patch_size)
    all_discarded: list[dict] = []

    has_source = False
    source_px: tuple[float, float] | None = None

    if gt_path is not None:
        has_source, ext_x, ext_y = read_source_info(gt_path)
        if has_source:
            source_px = source_pixel_in_dirty(gt_path, dirty_path, ext_x, ext_y)
            patches, discarded_source = filter_patches_by_source(
                patches,
                source_px[0],
                source_px[1],
                margin=source_margin,
            )
            all_discarded.extend(discarded_source)

    patch_stats = [compute_patch_stats(data, p, detrend=detrend) for p in patches]
    valid_stats, discarded_stats = filter_patches_by_std_consistency(
        patch_stats, factor=std_consistency_factor
    )
    all_discarded.extend(discarded_stats)

    if len(valid_stats) < 2:
        logging.warning(
            "Only %d valid noise patch(es) for %s (need >= 2). "
            "Using whatever patches survived.",
            len(valid_stats),
            dirty_path,
        )

    if not valid_stats:
        raise RuntimeError(
            f"No valid noise patches found for {dirty_path}. "
            "All patches were discarded by source overlap or std inconsistency."
        )

    sigma_noise = float(np.mean([s.std for s in valid_stats]))
    if not np.isfinite(sigma_noise) or sigma_noise <= 0.0:
        raise RuntimeError(
            f"Invalid sigma_noise={sigma_noise} for {dirty_path}. "
            "Check the dirty image and corner-patch selection."
        )

    threshold = multiplier * sigma_noise

    return NoiseResult(
        sigma_noise=sigma_noise,
        multiplier=multiplier,
        threshold=threshold,
        valid_patches=valid_stats,
        discarded_patches=all_discarded,
        source_px=source_px,
        has_point_source=has_source,
    )


def plot_noise_validation(
    dirty_path: str | Path,
    result: NoiseResult,
    out_png: str | Path,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> None:
    """Save a diagnostic PNG of the dirty image with patches and source marker."""
    try:
        import matplotlib
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plot_noise_validation") from exc

    data = fits.getdata(str(dirty_path)).squeeze().astype(np.float64)
    ny, nx = data.shape[-2], data.shape[-1]
    patches = define_corner_patches((ny, nx), patch_size=patch_size)
    valid_names = {p.name for p in result.valid_patches}

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data, origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax, label="Jy/beam")

    for patch in patches:
        color = "lime" if patch.name in valid_names else "red"
        rect = mpatches.Rectangle(
            (patch.col_slice.start, patch.row_slice.start),
            patch.col_slice.stop - patch.col_slice.start,
            patch.row_slice.stop - patch.row_slice.start,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            patch.col_slice.start + 0.5,
            patch.row_slice.start + 0.5,
            patch.name,
            color=color,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
        )

    if result.source_px is not None:
        ax.plot(
            result.source_px[1],
            result.source_px[0],
            marker="+",
            color="cyan",
            markersize=16,
            markeredgewidth=2,
            label="point source (header)",
        )
        ax.legend(loc="upper right")

    ax.set_title(
        f"sigma_noise={result.sigma_noise:.4e}  threshold(x{result.multiplier})="
        f"{result.threshold:.4e}"
    )
    ax.set_xlabel("pixel")
    ax.set_ylabel("pixel")

    fig.tight_layout()
    fig.savefig(str(out_png), dpi=150)
    plt.close(fig)


def log_noise_result(
    result: NoiseResult,
    log_func: callable = logging.info,
) -> None:
    """Log a NoiseResult in a structured, human-readable way."""
    log_func("[noise] HAS_EXT=%s", result.has_point_source)
    if result.source_px is not None:
        log_func("[noise] point source pixel=(row=%.2f, col=%.2f)", *result.source_px)

    for patch in result.valid_patches:
        log_func(
            "[noise] patch %s VALID: mean=%.6e std=%.6e rms=%.6e n=%d",
            patch.name, patch.mean, patch.std, patch.rms, patch.n,
        )
    for disc in result.discarded_patches:
        extra = ""
        if "mean" in disc and "std" in disc:
            extra = (
                f" mean={disc['mean']:.6e} std={disc['std']:.6e} "
                f"rms={disc['rms']:.6e}"
            )
        elif "std" in disc:
            extra = f" std={disc['std']:.6e}"
        log_func("[noise] patch %s DISCARDED: %s%s", disc["patch"], disc["reason"], extra)

    log_func(
        "[noise] sigma_noise=%.6e (mean std of %d valid patches)",
        result.sigma_noise, len(result.valid_patches),
    )
    log_func(
        "[noise] threshold (multiplier %.2f)=%.6e",
        result.multiplier, result.threshold,
    )
