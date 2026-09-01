#!/usr/bin/env python3
"""
Plot a 2x3 comparison figure for a simulated RI-Benchmark target,
showing the ground truth, dirty image, and the best-configuration
reconstructions of CLEAN (CASA), WSClean, Wavelet-based (DeepInverse)
and R2D2 (custom) with their PSNR and SSIM.

The 5 comparable panels (Ground Truth, CLEAN, WSClean, Wavelet-based,
R2D2) share a single color scale (vmin=0, vmax=common max). The
Dirty Image panel keeps its own scale.

The default target (disk_s4_r2_c1_x1_001) was chosen from the sweep results
because it has a high PSNR and clearly illustrates the overall trend
reported in results/sweep/sweep_results.txt:
    DeepInverse > R2D2 >> CLEAN ~ WSClean

The PSNR/SSIM values are taken from results/sweep/sweep_state.csv for the
best global configurations:
    casa:     g0p1_t0p002
    wsclean:  g0p1_t0p005
    deepinv:  l0p1_i50
    r2d2:     k5

Usage:
    .envs/.benchmark/bin/python src/utils/plot_comparison_with_psnr.py
    .envs/.benchmark/bin/python src/utils/plot_comparison_with_psnr.py \
        --target disk_s4_r2_c1_x1_001 --output my_comparison.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.patches import Ellipse
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# Use the bundled Computer Modern Roman font (cmr10) shipped with matplotlib.
# The generic "Computer Modern" name is not installed system-wide, but cmr10 is
# the same design and is available as a bundled TTF.
plt.rcParams["font.family"] = "cmr10"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["mathtext.fontset"] = "cm"


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = "disk_s4_r2_c1_x1_001"

# Best configs per method as identified in results/sweep/sweep_results.txt
BEST_CONFIGS = {
    "casa": "g0p1_t0p002",
    "wsclean": "g0p1_t0p005",
    "deepinv": "l0p1_i50",
    "r2d2": "k5",
}

# Exact panel titles requested by the user
DISPLAY_NAMES = {
    "groundtruth": "Ground Truth",
    "dirty": "Dirty Image",
    "casa": "CLEAN (CASA)",
    "wsclean": "WSClean",
    "deepinv": "Wavelet-based\n(DeepInverse)",
    "r2d2": "R2D2 (custom)",
}


def load_fits(path):
    """Load a FITS file and squeeze it to 2D."""
    data = fits.getdata(path).squeeze().astype(np.float64)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D image after squeeze, got shape {data.shape}")
    return data


def find_sweep_metric(target, method, config_id, metric):
    """Read a metric (psnr or ssim) for a target/method/config from sweep_state.csv."""
    csv_path = BASE_DIR / "results" / "sweep" / "sweep_state.csv"
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row["target"] == target and
                    row["method"] == method and
                    row["config_id"] == config_id and
                    row["status"] == "completed"):
                return float(row[metric])
    raise ValueError(f"No completed sweep row for {target}/{method}/{config_id}/{metric}")


def compute_image_psnr(reconstruction, ground_truth):
    """Compute PSNR following RI-Benchmark/src/compute_metrics.py logic."""
    reconstruction = np.clip(reconstruction, 0, None)
    data_range = float(ground_truth.max() - ground_truth.min())
    return compute_psnr(reconstruction, ground_truth, data_range=data_range)


def compute_image_ssim(reconstruction, ground_truth):
    """Compute SSIM following RI-Benchmark/src/compute_metrics.py logic."""
    reconstruction = np.clip(reconstruction, 0, None)
    data_range = float(ground_truth.max() - ground_truth.min())
    return compute_ssim(reconstruction, ground_truth, data_range=data_range)


def draw_synthesized_beam(ax, fits_path):
    """Draw the synthesized beam from the FITS header in the bottom-left corner.

    Reads BMAJ/BMIN/BPA (degrees) and CDELT1/CDELT2 (degrees/pixel) and draws
    a white ellipse representing the restoring beam.
    """
    hdr = fits.getheader(fits_path)
    if "BMAJ" not in hdr or "BMIN" not in hdr or "BPA" not in hdr:
        return False

    bmaj_deg = float(hdr["BMAJ"])
    bmin_deg = float(hdr["BMIN"])
    bpa_deg = float(hdr["BPA"])

    cdelt1 = float(hdr.get("CDELT1", 1.0))
    cdelt2 = float(hdr.get("CDELT2", 1.0))

    # Beam FWHM in pixels
    bmaj_pix = bmaj_deg / abs(cdelt2)
    bmin_pix = bmin_deg / abs(cdelt1)

    # Convert BPA (North through East) to matplotlib angle (CCW from +x).
    # North pixel vector: (0, sign(cdelt2)); East pixel vector: (sign(cdelt1), 0).
    rad = np.deg2rad(bpa_deg)
    pixel_x = np.sign(cdelt1) * np.sin(rad)
    pixel_y = np.sign(cdelt2) * np.cos(rad)
    angle_deg = np.degrees(np.arctan2(pixel_y, pixel_x))

    # Place ellipse in the bottom-right corner, with some padding from the edge
    ny, nx = ax.get_images()[0].get_array().shape[:2]
    center_x = 0.94 * nx
    center_y = 0.04 * ny

    # Major axis is drawn as the width, minor as height; rotate to BPA
    ellipse = Ellipse(
        xy=(center_x, center_y),
        width=2 * bmaj_pix,
        height=2 * bmin_pix,
        angle=angle_deg,
        edgecolor="white",
        facecolor="none",
        linewidth=1.0,
        alpha=0.9,
        zorder=10,
    )
    ax.add_patch(ellipse)

    # Optional: small cross hairs to mark the beam center
    ax.plot(
        center_x,
        center_y,
        marker="+",
        markersize=4,
        color="white",
        alpha=0.7,
        zorder=11,
    )
    return True


def get_image_paths(target):
    """Resolve the FITS paths for all six panels."""
    exp_dir = BASE_DIR / "experiments" / "simulated" / target
    data_dir = BASE_DIR / "data" / "simulated" / target

    return {
        "groundtruth": data_dir / f"{target}.fits",
        "dirty": (
            exp_dir / f"{target}_casa" / f"{target}_casa_g0p1_t0p002" /
            f"{target}_casa_g0p1_t0p002_dirty.fits"
        ),
        "casa": (
            exp_dir / f"{target}_casa" / f"{target}_casa_{BEST_CONFIGS['casa']}" /
            f"{target}_casa_{BEST_CONFIGS['casa']}.fits"
        ),
        "wsclean": (
            exp_dir / f"{target}_wsclean" /
            f"{target}_wsclean_{BEST_CONFIGS['wsclean']}" /
            f"{target}_wsclean_{BEST_CONFIGS['wsclean']}.fits"
        ),
        "deepinv": (
            exp_dir / f"{target}_deepinv" /
            f"{target}_deepinv_{BEST_CONFIGS['deepinv']}" /
            f"{target}_deepinv_{BEST_CONFIGS['deepinv']}.fits"
        ),
        "r2d2": (
            exp_dir / f"{target}_r2d2" / f"{target}_r2d2_{BEST_CONFIGS['r2d2']}" /
            f"{target}_r2d2.fits"
        ),
    }


def plot_comparison(target, output_path=None):
    """Generate and save the comparison figure."""
    paths = get_image_paths(target)

    # Load images
    images = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        images[name] = load_fits(path)

    gt = images["groundtruth"]

    # PSNRs, SSIMs and imaging times: prefer sweep_state.csv values;
    # PSNR/SSIM fall back to recomputation if the sweep row is missing.
    psnrs = {}
    ssims = {}
    times = {}
    for method in ["casa", "wsclean", "deepinv", "r2d2"]:
        try:
            psnrs[method] = find_sweep_metric(target, method, BEST_CONFIGS[method], "psnr")
            ssims[method] = find_sweep_metric(target, method, BEST_CONFIGS[method], "ssim")
            times[method] = find_sweep_metric(target, method, BEST_CONFIGS[method], "t")
        except ValueError:
            psnrs[method] = compute_image_psnr(images[method], gt)
            ssims[method] = compute_image_ssim(images[method], gt)
            times[method] = None

    # Convert the dirty image from Jy/beam to Jy/pixel using its FITS header
    dirty_hdr = fits.getheader(paths["dirty"])
    if dirty_hdr.get("BUNIT", "").strip().lower() == "jy/beam":
        bmaj = float(dirty_hdr["BMAJ"])
        bmin = float(dirty_hdr["BMIN"])
        cdelt1 = float(dirty_hdr.get("CDELT1", 1.0))
        cdelt2 = float(dirty_hdr.get("CDELT2", 1.0))
        # Beam area in pixels: Omega_beam / Omega_pixel
        beam_area_pix = (
            (np.pi * bmaj * bmin)
            / (4.0 * np.log(2.0) * abs(cdelt1 * cdelt2))
        )
        images["dirty"] = images["dirty"] / beam_area_pix

    # Shared color scale for the 5 comparable panels (exclude dirty image)
    comparable_keys = ["groundtruth", "casa", "wsclean", "deepinv", "r2d2"]
    shared_vmax = max(images[k].max() for k in comparable_keys)

    # Layout: 3 rows x 2 columns
    # Top row: groundtruth, dirty image
    # Middle row: casa, wsclean
    # Bottom row: deepinv, r2d2
    layout = [
        ["groundtruth", "dirty"],
        ["casa", "wsclean"],
        ["deepinv", "r2d2"],
    ]

    # Tall figure, readable at 300 dpi
    fig, axes = plt.subplots(
        3, 2, figsize=(5.5, 6.5), facecolor="black", constrained_layout=False
    )
    plt.subplots_adjust(
        left=0.02, right=0.94, top=0.96, bottom=0.02, hspace=0.02, wspace=0.08
    )

    for row_idx, row_keys in enumerate(layout):
        for col_idx, key in enumerate(row_keys):
            ax = axes[row_idx, col_idx]
            img = images[key]

            # Shared scale for comparable panels; dirty image keeps its own scale
            if key in comparable_keys:
                vmax = shared_vmax
            else:
                vmax = img.max()

            im = ax.imshow(
                img,
                cmap="inferno",
                origin="lower",
                vmin=0,
                vmax=vmax,
                interpolation="nearest",
            )

            # Panel title / labels at the top-left
            display_name = DISPLAY_NAMES[key]
            ax.text(
                0.03,
                0.98,
                display_name,
                transform=ax.transAxes,
                fontsize=10,
                color="white",
                fontweight="bold",
                va="top",
                ha="left",
            )

            # Metrics for the four method panels:
            #   title at upper-left, all metrics (T, PSNR, SSIM) at bottom-left
            if key in ["casa", "wsclean", "deepinv", "r2d2"]:
                t_value = times[key]
                t_str = f"T: {t_value:.2f} s" if t_value is not None else "T: —"
                metric_text = (
                    f"{t_str}\n"
                    f"PSNR: {psnrs[key]:.2f} dB\n"
                    f"SSIM: {ssims[key]:.4f}"
                )
                ax.text(
                    0.03,
                    0.10,
                    metric_text,
                    transform=ax.transAxes,
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                    va="bottom",
                    ha="left",
                    linespacing=1.0,
                )

            ax.axis("off")

            # Colorbar with white ticks/outlines on the black figure
            cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cbar.ax.tick_params(labelsize=9, colors="white")
            cbar.outline.set_edgecolor("white")
            # Vertical tick labels to save horizontal space
            for label in cbar.ax.get_yticklabels():
                label.set_rotation(90)
                label.set_va("center")

            # Draw the real synthesized beam (from FITS header) only for
            # CLEAN (CASA) and WSClean, which are the methods that produce
            # a meaningful restoring beam in their FITS outputs.
            if key in ["casa", "wsclean"]:
                draw_synthesized_beam(ax, paths[key])

    if output_path is None:
        out_dir = BASE_DIR / "results" / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{target}_comparison_with_psnr.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, facecolor="black", edgecolor="none")
    print(f"Saved comparison figure to: {output_path}")
    print("Metrics shown (best sweep config):")
    for method in ["casa", "wsclean", "deepinv", "r2d2"]:
        t_str = f"{times[method]:.2f}s" if times[method] is not None else "—"
        print(
            f"  {method:10s} {BEST_CONFIGS[method]:10s} = "
            f"PSNR {psnrs[method]:.2f} dB, SSIM {ssims[method]:.4f}, T {t_str}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Create a 6-panel comparison figure with PSNR/SSIM labels."
    )
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help=f"Target name (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path (default: results/images/<target>_comparison_with_psnr.png)",
    )
    args = parser.parse_args()

    plot_comparison(args.target, args.output)


if __name__ == "__main__":
    main()
