#!/usr/bin/env python3
"""
Generate a 1x2 comparison figure for the archival HD163296 continuum image
and its R2D2 reconstruction.

The DSHARP FITS is first cropped to a square centered on the source, then
resized to 64x64 so that the source has the same angular size as the 64x64
R2D2 reconstruction. The resulting 64x64 FITS is saved separately; the
original is never modified.

Outputs:
- results/images/HD163296_continuum_cropped64.fits
- results/images/HD163296_continuum_r2d2_comparison.png

Usage:
    .envs/.benchmark/bin/python src/utils/plot_archival_r2d2.py
    .envs/.benchmark/bin/python src/utils/plot_archival_r2d2.py --zoom-factor 1.5
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from skimage.transform import resize

# Use the bundled Computer Modern Roman font (cmr10) shipped with matplotlib.
plt.rcParams["font.family"] = "cmr10"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["mathtext.fontset"] = "cm"


BASE_DIR = Path(__file__).resolve().parents[2]

DEFAULT_ARCHIVAL = BASE_DIR / "data" / "archival" / "HD163296_continuum.fits"
DEFAULT_R2D2 = (
    BASE_DIR
    / "experiments"
    / "archival"
    / "HD163296_continuum"
    / "HD163296_continuum_r2d2"
    / "HD163296_continuum_r2d2.fits"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "images"


def load_fits(path):
    """Load a FITS file and squeeze it to 2D."""
    data = fits.getdata(path).squeeze().astype(np.float64)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D image after squeeze, got shape {data.shape}")
    return data


def jybeam_to_jypixel(data, hdr):
    """Convert an image from Jy/beam to Jy/pixel using the beam in the header."""
    if hdr.get("BUNIT", "").strip().lower() != "jy/beam":
        return data
    if "BMAJ" not in hdr or "BMIN" not in hdr:
        raise ValueError("Header is Jy/beam but missing BMAJ/BMIN")
    cdelt = abs(float(hdr["CDELT1"]))
    beam_area = np.pi * float(hdr["BMAJ"]) * float(hdr["BMIN"]) / (4.0 * np.log(2.0) * cdelt**2)
    return data / beam_area


def parse_imaging_time(log_path):
    """Extract the imaging time in seconds from a method log file."""
    if not log_path.exists():
        return None
    text = log_path.read_text()

    # R2D2 log: "Imaging finished in 4/5 iterations in 1.96533 sec"
    r2d2_match = re.search(r"Imaging finished in \d+/\d+ iterations in\s+([0-9.eE+-]+)\s*sec", text)
    if r2d2_match:
        return float(r2d2_match.group(1))

    return None


def crop_and_downsample(archival_path, r2d2_path, zoom_factor=1.0):
    """Center-crop the DSHARP image and resize to 64x64."""
    archival_data = fits.getdata(archival_path).squeeze().astype(np.float64)
    archival_hdr = fits.getheader(archival_path)
    r2d2_hdr = fits.getheader(r2d2_path)

    ny, nx = archival_data.shape
    r2d2_nx = int(r2d2_hdr["NAXIS1"])
    r2d2_ny = int(r2d2_hdr["NAXIS2"])

    # Angular pixel scales (absolute values, degrees per pixel)
    arch_cdelt_x = abs(float(archival_hdr["CDELT1"]))
    arch_cdelt_y = abs(float(archival_hdr["CDELT2"]))
    r2d2_cdelt_x = abs(float(r2d2_hdr["CDELT1"]))
    r2d2_cdelt_y = abs(float(r2d2_hdr["CDELT2"]))

    # How many DSHARP pixels cover the same FOV as the R2D2 image?
    # zoom_factor > 1 makes the crop larger, i.e. less zoom.
    crop_nx = int(round(zoom_factor * r2d2_nx * r2d2_cdelt_x / arch_cdelt_x))
    crop_ny = int(round(zoom_factor * r2d2_ny * r2d2_cdelt_y / arch_cdelt_y))
    crop_nx = min(crop_nx, nx)
    crop_ny = min(crop_ny, ny)

    # Center crop on the source (CRPIX is 1-indexed in FITS)
    crpix_x = float(archival_hdr.get("CRPIX1", nx / 2.0 + 0.5))
    crpix_y = float(archival_hdr.get("CRPIX2", ny / 2.0 + 0.5))
    center_x = int(round(crpix_x - 1))  # to 0-based
    center_y = int(round(crpix_y - 1))

    start_x = max(0, center_x - crop_nx // 2)
    start_y = max(0, center_y - crop_ny // 2)
    end_x = min(nx, start_x + crop_nx)
    end_y = min(ny, start_y + crop_ny)
    # Recenter if the crop hit an edge
    if end_x - start_x < crop_nx:
        start_x = nx - crop_nx
    if end_y - start_y < crop_ny:
        start_y = ny - crop_ny

    cropped = archival_data[start_y:end_y, start_x:end_x]

    # Convert to Jy/pixel with the original DSHARP beam before changing the pixel scale
    cropped_jypix = jybeam_to_jypixel(cropped, archival_hdr)

    # Nearest-neighbour resize (pure pixels), scaled by pixel-area ratio to keep
    # the total flux per new pixel in Jy/pixel
    downsampled = resize(
        cropped_jypix,
        (r2d2_ny, r2d2_nx),
        order=0,
        mode="constant",
        cval=0.0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.float64)
    area_ratio = (r2d2_nx * r2d2_ny) / (crop_nx * crop_ny)
    downsampled_jypix = downsampled / area_ratio

    new_hdr = archival_hdr.copy()
    new_hdr["NAXIS"] = 2
    new_hdr["NAXIS1"] = r2d2_nx
    new_hdr["NAXIS2"] = r2d2_ny
    new_hdr["CDELT1"] = crop_nx * arch_cdelt_x * np.sign(float(archival_hdr["CDELT1"])) / r2d2_nx
    new_hdr["CDELT2"] = crop_ny * arch_cdelt_y * np.sign(float(archival_hdr["CDELT2"])) / r2d2_ny
    new_hdr["CRPIX1"] = r2d2_nx / 2.0
    new_hdr["CRPIX2"] = r2d2_ny / 2.0
    new_hdr["BUNIT"] = "Jy/pixel"
    for key in ["NAXIS3", "NAXIS4", "CRPIX3", "CRPIX4"]:
        if key in new_hdr:
            del new_hdr[key]

    return downsampled_jypix, archival_hdr, r2d2_hdr, new_hdr, crop_nx, crop_ny


def plot_comparison(
    archival_data,
    r2d2_data,
    r2d2_time=None,
    output_dir=DEFAULT_OUTPUT_DIR,
    cropped_fits_dir=DEFAULT_OUTPUT_DIR,
    cropped_hdr=None,
):
    """Create and save the 1x2 comparison figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cropped_fits_dir = Path(cropped_fits_dir)
    cropped_fits_dir.mkdir(parents=True, exist_ok=True)

    cropped_path = cropped_fits_dir / "HD163296_continuum_cropped64.fits"
    fits.writeto(cropped_path, archival_data, header=cropped_hdr, overwrite=True)
    print(f"Saved cropped archival FITS: {cropped_path}")

    fig, axes = plt.subplots(
        1, 2, figsize=(6.0, 3.3), facecolor="black", constrained_layout=False
    )
    plt.subplots_adjust(
        left=0.03, right=0.94, top=0.96, bottom=0.02, wspace=0.10
    )

    panels = [
        ("HD163296 continuum\n(archival, 64x64)", archival_data, None),
        ("R2D2 reconstruction", r2d2_data, r2d2_time),
    ]

    for ax, (title, img, time) in zip(axes, panels):
        im = ax.imshow(
            img,
            cmap="inferno",
            origin="lower",
            vmin=0,
            vmax=img.max(),
            interpolation="nearest",
        )

        # Title at upper-left
        ax.text(
            0.03,
            0.98,
            title,
            transform=ax.transAxes,
            fontsize=10,
            color="white",
            fontweight="bold",
            va="top",
            ha="left",
        )

        # Imaging time at bottom-left
        if time is not None:
            ax.text(
                0.03,
                0.10,
                f"T: {time:.2f} s",
                transform=ax.transAxes,
                fontsize=9,
                color="white",
                fontweight="bold",
                va="bottom",
                ha="left",
            )

        ax.axis("off")

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.ax.tick_params(labelsize=7, colors="white")
        cbar.outline.set_edgecolor("white")
        for label in cbar.ax.get_yticklabels():
            label.set_rotation(90)
            label.set_va("center")

    output_path = output_dir / "HD163296_continuum_r2d2_comparison.png"
    plt.savefig(output_path, dpi=300, facecolor="black", edgecolor="none")
    print(f"Saved comparison figure: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a 1x2 comparison of archival HD163296 and R2D2."
    )
    parser.add_argument(
        "--archival",
        type=str,
        default=str(DEFAULT_ARCHIVAL),
        help="Path to the archival FITS file",
    )
    parser.add_argument(
        "--r2d2",
        type=str,
        default=str(DEFAULT_R2D2),
        help="Path to the R2D2 reconstruction FITS file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the output PNG and cropped FITS",
    )
    parser.add_argument(
        "--zoom-factor",
        type=float,
        default=1.0,
        help="Zoom-out factor for the archival panel (>1 = more FOV, default 1.0)",
    )
    args = parser.parse_args()

    archival_data, archival_hdr, r2d2_hdr, target_hdr, crop_nx, crop_ny = crop_and_downsample(
        args.archival, args.r2d2, zoom_factor=args.zoom_factor
    )
    r2d2_data = load_fits(args.r2d2)

    print(f"Cropped DSHARP from {fits.getdata(args.archival).squeeze().shape} "
          f"to {crop_ny}x{crop_nx}, then resized to 64x64")

    r2d2_log = Path(args.r2d2).with_suffix(".log")
    r2d2_time = parse_imaging_time(r2d2_log)

    plot_comparison(
        archival_data,
        r2d2_data,
        r2d2_time=r2d2_time,
        output_dir=args.output_dir,
        cropped_fits_dir=args.output_dir,
        cropped_hdr=target_hdr,
    )


if __name__ == "__main__":
    main()
