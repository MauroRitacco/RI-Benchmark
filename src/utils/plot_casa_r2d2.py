#!/usr/bin/env python3
"""
Generate a 1x2 comparison figure for the CASA and R2D2 reconstructions of the
archival HD163296 continuum target.

Outputs:
- results/images/HD163296_continuum_casa_r2d2_comparison.png

Usage:
    .envs/.benchmark/bin/python src/utils/plot_casa_r2d2.py
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

# Use the bundled Computer Modern Roman font (cmr10) shipped with matplotlib.
plt.rcParams["font.family"] = "cmr10"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["mathtext.fontset"] = "cm"


BASE_DIR = Path(__file__).resolve().parents[2]

DEFAULT_CASA = (
    BASE_DIR
    / "experiments"
    / "archival"
    / "HD163296_continuum"
    / "HD163296_continuum_casa"
    / "HD163296_continuum_casa.fits"
)
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


def parse_imaging_time(log_path):
    """Extract the imaging time in seconds from a method log file."""
    if not log_path.exists():
        return None
    text = log_path.read_text()

    # CASA log: "Imaging time: 1.97 seconds"
    casa_match = re.search(r"Imaging time:\s*([0-9.eE+-]+)\s*seconds", text)
    if casa_match:
        return float(casa_match.group(1))

    # R2D2 log: "Imaging finished in 4/5 iterations in 1.96533 sec"
    r2d2_match = re.search(r"Imaging finished in \d+/\d+ iterations in\s+([0-9.eE+-]+)\s*sec", text)
    if r2d2_match:
        return float(r2d2_match.group(1))

    return None


def plot_comparison(casa_data, r2d2_data, casa_time=None, r2d2_time=None, output_dir=DEFAULT_OUTPUT_DIR):
    """Create and save the 2x1 comparison figure (CASA on top, R2D2 below)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2, 1, figsize=(3.5, 6.5), facecolor="black", constrained_layout=False
    )
    plt.subplots_adjust(
        left=0.03, right=0.94, top=0.96, bottom=0.02, hspace=0.02
    )

    panels = [
        ("CLEAN (CASA)", casa_data, casa_time),
        ("R2D2", r2d2_data, r2d2_time),
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

        # Title at upper-left, matching reference font size
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

        # Imaging time at bottom-left, matching reference metric style
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

    output_path = output_dir / "HD163296_continuum_casa_r2d2_comparison.png"
    plt.savefig(output_path, dpi=300, facecolor="black", edgecolor="none")
    print(f"Saved comparison figure: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a 1x2 comparison of CASA and R2D2 for HD163296."
    )
    parser.add_argument(
        "--casa",
        type=str,
        default=str(DEFAULT_CASA),
        help="Path to the CASA reconstruction FITS file",
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
        help="Directory for the output PNG",
    )
    args = parser.parse_args()

    casa_data = load_fits(args.casa)
    r2d2_data = load_fits(args.r2d2)

    casa_path = Path(args.casa)
    r2d2_path = Path(args.r2d2)

    casa_log = casa_path.with_suffix(".log")
    r2d2_log = r2d2_path.with_suffix(".log")

    casa_time = parse_imaging_time(casa_log)
    r2d2_time = parse_imaging_time(r2d2_log)

    plot_comparison(casa_data, r2d2_data, casa_time, r2d2_time, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
