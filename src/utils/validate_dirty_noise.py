#!/usr/bin/env python3
"""Standalone validation / characterization of dirty-image noise estimation.

Default: 4 targets. --characterize: 12-target study with CSV + sigma vs tau plot.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.io import loadmat

# Suppress astropy FITS header warnings during validation
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*OBSGEO.*")

# Allow imports from the project root
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from src.utils.dirty_noise import (
    DEFAULT_PATCH_SIZE,
    DEFAULT_SOURCE_MARGIN,
    DEFAULT_STD_CONSISTENCY_FACTOR,
    compute_patch_stats,
    define_corner_patches,
    filter_patches_by_source,
    filter_patches_by_std_consistency,
    log_noise_result,
    plot_noise_validation,
    read_source_info,
    source_pixel_in_dirty,
)
from src.utils.dirty_noise import estimate_noise as _estimate_noise


# Configure a clean single handler for the root logger
logging.basicConfig(level=logging.WARNING, format="%(message)s")
_root_logger = logging.getLogger()
for h in list(_root_logger.handlers):
    _root_logger.removeHandler(h)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(message)s"))
_root_logger.addHandler(_h)
_root_logger.setLevel(logging.WARNING)


VALIDATION_TARGETS = [
    "disk_s0p5_r1_c0_x0_005",   # small disk, no source
    "disk_s4_r4_c1_x0_005",     # large disk (4 arcsec), 4 rings, no source
    "disk_s0p5_r1_c0_x1_005",   # small disk, with point source
    "disk_s0p5_r2_c1_x1_005",   # small disk, with point source AND star
]

CHARACTERIZATION_TARGETS = [
    # s0p5
    "disk_s0p5_r1_c0_x0_005",
    "disk_s0p5_r4_c0_x1_005",
    "disk_s0p5_r1_c0_x1_005",
    "disk_s0p5_r2_c1_x1_005",
    # s2
    "disk_s2_r1_c0_x0_005",
    "disk_s2_r4_c0_x1_005",
    "disk_s2_r2_c1_x1_005",
    "disk_s2_r4_c1_x0_005",
    # s4
    "disk_s4_r1_c0_x0_005",
    "disk_s4_r4_c0_x1_005",
    "disk_s4_r2_c0_x1_005",
    "disk_s4_r4_c1_x0_005",
]


def target_paths(target: str, base_dir: Path, dirty_dir: Path | None = None):
    """Return ground truth and dirty paths for a target.

    If dirty_dir is given, look for <target>_casa_dirty_dirty.fits there.
    Otherwise fall back to the existing <target>_res.fits in data/simulated/.
    """
    category = "simulated" if (base_dir / "data" / "simulated" / target).exists() else "archival"
    gt_path = base_dir / "data" / category / target / f"{target}.fits"
    if dirty_dir is not None:
        dirty_path = dirty_dir / target / f"{target}_casa_dirty_dirty.fits"
    else:
        dirty_path = base_dir / "data" / category / target / f"{target}_res.fits"
    return gt_path, dirty_path


def parse_target_factors(target: str) -> dict[str, str]:
    """Parse the sweep target name into size/rings/center/extra factors."""
    # disk_s0p5_r1_c0_x1_005 -> size=0p5, rings=1, center=0, extra=1
    parts = target.split("_")
    return {
        "size": parts[1].replace("s", ""),
        "rings": parts[2].replace("r", ""),
        "center": parts[3].replace("c", ""),
        "extra": parts[4].replace("x", ""),
        "rep": parts[5],
    }


def read_tau(mat_path: Path) -> float:
    """Return the visibility noise std tau from a measurement .mat file."""
    mat = loadmat(str(mat_path))
    nW = mat["nW"].ravel()
    return float(1.0 / nW[0])


def characterize_targets(args: argparse.Namespace) -> None:
    """Run the 12-target characterization and produce table + plots."""
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    per_target_details: list[tuple[str, list]] = []

    for target in args.targets:
        gt_path, dirty_path = target_paths(target, args.base_dir, args.dirty_dir)
        factors = parse_target_factors(target)
        mat_path = gt_path.with_suffix(".mat")

        if not gt_path.exists() or not dirty_path.exists() or not mat_path.exists():
            print(f"[skip] {target}: missing files")
            continue

        data = fits.getdata(str(dirty_path)).squeeze().astype(np.float64)
        patches = define_corner_patches(data.shape, patch_size=DEFAULT_PATCH_SIZE)

        has_source, ext_x, ext_y = read_source_info(gt_path)
        source_px = None
        if has_source:
            source_px = source_pixel_in_dirty(gt_path, dirty_path, ext_x, ext_y)
            patches, _ = filter_patches_by_source(
                patches, source_px[0], source_px[1], margin=DEFAULT_SOURCE_MARGIN
            )

        # Compute sigma with and without 2D plane detrending.
        results: dict[bool, tuple] = {}
        for detrend in (False, True):
            patch_stats = [
                compute_patch_stats(data, p, detrend=detrend) for p in patches
            ]
            valid_stats, _ = filter_patches_by_std_consistency(
                patch_stats, factor=DEFAULT_STD_CONSISTENCY_FACTOR
            )
            sigma_noise = (
                float(np.mean([s.std for s in valid_stats])) if valid_stats else np.nan
            )
            results[detrend] = (patch_stats, valid_stats, sigma_noise)

        tau = read_tau(mat_path)

        no_stats, no_valid, sigma_no = results[False]
        yes_stats, yes_valid, sigma_yes = results[True]

        row = {
            "target": target,
            "size": factors["size"],
            "rings": factors["rings"],
            "center": factors["center"],
            "extra": factors["extra"],
            "tau": tau,
            "sigma_no_detrend": sigma_no,
            "sigma_detrend": sigma_yes,
            "n_patches_source_pass": len(yes_stats),
            "n_patches_std_consistent": len(yes_valid),
        }
        # Report detrended patch stats in columns.
        for s in yes_stats:
            row[f"{s.name}_mean"] = s.mean
            row[f"{s.name}_std"] = s.std
            row[f"{s.name}_rms"] = s.rms
            row[f"{s.name}_source_pass"] = True
            row[f"{s.name}_std_consistent"] = any(v.name == s.name for v in yes_valid)
        rows.append(row)
        per_target_details.append((target, yes_stats, yes_valid))

    csv_path = args.out_dir / "characterization.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV saved: {csv_path}")

    print("\n" + "=" * 120)
    print("Characterization table (new criterion: std <= 2 * median std)")
    print("=" * 120)
    print(
        f"{'target':<25} {'size':>5} {'r':>3} {'c':>3} {'x':>3} "
        f"{'src_ok':>7} {'std_ok':>7} {'sigma_no_det':>16} {'sigma_detrend':>16} "
        f"{'patch stds':>60}"
    )
    for r in rows:
        stds_str = ", ".join(
            f"{name}={r[f'{name}_std']:.3f}"
            + ("*" if not r[f"{name}_std_consistent"] else "")
            for name in ["TL", "TR", "BL", "BR"]
            if f"{name}_std" in r
        )
        sigma_no_str = f"{r['sigma_no_detrend']:.4e}" if np.isfinite(r["sigma_no_detrend"]) else "FAILED"
        sigma_yes_str = f"{r['sigma_detrend']:.4e}" if np.isfinite(r["sigma_detrend"]) else "FAILED"
        print(
            f"{r['target']:<25} {r['size']:>5} {r['rings']:>3} {r['center']:>3} {r['extra']:>3} "
            f"{r['n_patches_source_pass']:>7} {r['n_patches_std_consistent']:>7} "
            f"{sigma_no_str:>16} {sigma_yes_str:>16} {stds_str:>60}"
        )
    print("\n(*) asterisk: parche descartado por std > 2 * mediana")

    print("\n" + "=" * 120)
    print("Per-patch details")
    print("=" * 120)
    print(
        f"{'target':<25} {'patch':>6} {'mean':>12} {'std':>12} {'rms':>12} "
        f"{'valid?':>8}"
    )
    for target, patch_stats, valid_stats in per_target_details:
        valid_names = {v.name for v in valid_stats}
        for s in patch_stats:
            mark = "YES" if s.name in valid_names else "NO*"
            print(
                f"{target:<25} {s.name:>6} {s.mean:>12.4e} {s.std:>12.4e} "
                f"{s.rms:>12.4e} {mark:>8}"
            )

    _plot_sigma_vs_tau(rows, args.out_dir / "sigma_vs_tau.png")

    _print_analysis(rows)


def _plot_sigma_vs_tau(rows: list[dict], out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping scatter plot")
        return

    size_colors = {"0p5": "tab:blue", "2": "tab:orange", "4": "tab:green"}
    fig, ax = plt.subplots(figsize=(7, 6))

    for size in ["0p5", "2", "4"]:
        xs = [r["tau"] for r in rows if r["size"] == size and np.isfinite(r["sigma_detrend"])]
        ys = [r["sigma_detrend"] for r in rows if r["size"] == size and np.isfinite(r["sigma_detrend"])]
        ax.scatter(xs, ys, c=size_colors[size], label=f"FOV {size} arcsec", s=80)

    # Diagonal y=x for reference
    all_vals = [v for r in rows for v in (r["tau"], r["sigma_detrend"]) if np.isfinite(v)]
    if all_vals:
        lims = (min(all_vals) * 0.8, max(all_vals) * 1.2)
        ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    ax.set_xlabel("tau (visibility noise std, Jy)")
    ax.set_ylabel("sigma_noise from corner std (Jy/beam)")
    ax.set_title("Dirty-image corner noise vs visibility noise")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=150)
    plt.close(fig)
    print(f"\nScatter plot saved: {out_png}")


def _print_analysis(rows: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("Analysis")
    print("=" * 80)

    # 1. Pedestal (|mean|/std) vs FOV
    print("\n1. Sidelope pedestal: |mean| / std per corner patch by FOV")
    for size in ["0p5", "2", "4"]:
        ratios = []
        for r in rows:
            if r["size"] != size:
                continue
            for name in ["TL", "TR", "BL", "BR"]:
                if f"{name}_mean" in r and f"{name}_std" in r:
                    std = r[f"{name}_std"]
                    if std > 0:
                        ratios.append(abs(r[f"{name}_mean"]) / std)
        if ratios:
            print(
                f"  FOV {size}: median |mean|/std = {np.median(ratios):.2f}, "
                f"max = {np.max(ratios):.2f} (N={len(ratios)} patches)"
            )

    # 2. Correlation sigma_noise vs tau
    taus = np.array([r["tau"] for r in rows])
    sigmas = np.array([r["sigma_detrend"] for r in rows])
    valid = np.isfinite(taus) & np.isfinite(sigmas) & (taus > 0) & (sigmas > 0)
    if valid.sum() > 1:
        log_tau = np.log10(taus[valid])
        log_sigma = np.log10(sigmas[valid])
        corr = np.corrcoef(log_tau, log_sigma)[0, 1]
        print(f"\n2. Correlation log10(sigma_noise) vs log10(tau): r = {corr:.3f}")
    else:
        print("\n2. Not enough valid points to compute correlation.")

    # 3. Targets with <2 valid patches under new std-consistency criterion
    print("\n3. Targets with <2 patches passing std-consistency (2x median) by FOV:")
    for size in ["0p5", "2", "4"]:
        n_total = sum(1 for r in rows if r["size"] == size)
        n_fail = sum(
            1 for r in rows if r["size"] == size and r["n_patches_std_consistent"] < 2
        )
        print(f"  FOV {size}: {n_fail}/{n_total} targets with <2 valid patches")

    n_fail_total = sum(1 for r in rows if r["n_patches_std_consistent"] < 2)
    print(f"\nTotal targets with <2 valid patches: {n_fail_total}/{len(rows)}")


def validate_targets(args: argparse.Namespace) -> None:
    """Default mode: validate noise estimation on selected targets."""
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Dirty-image noise estimation validation")
    print("=" * 80)

    summary: list[tuple[str, int, float | None]] = []

    for target in args.targets:
        gt_path, dirty_path = target_paths(target, args.base_dir, args.dirty_dir)

        lines: list[str] = []
        lines.append(f"\n--- Target: {target} ---")

        if not gt_path.exists():
            lines.append(f"[skip] ground truth not found: {gt_path}")
            _print_block(lines)
            summary.append((target, 0, None))
            continue
        if not dirty_path.exists():
            lines.append(f"[skip] dirty image not found: {dirty_path}")
            _print_block(lines)
            summary.append((target, 0, None))
            continue

        try:
            result = _estimate_noise(
                dirty_path=dirty_path,
                gt_path=gt_path,
                multiplier=1.0,
            )
        except Exception as exc:
            lines.append(f"[error] {target}: {exc}")
            _print_block(lines)
            summary.append((target, 0, None))
            continue

        logbuf = io.StringIO()
        h = logging.StreamHandler(logbuf)
        h.setFormatter(logging.Formatter("%(message)s"))
        tmp_logger = logging.getLogger(f"_validate_{target}")
        tmp_logger.handlers = []
        tmp_logger.addHandler(h)
        tmp_logger.setLevel(logging.INFO)

        log_noise_result(result, log_func=tmp_logger.info)

        lines.extend(logbuf.getvalue().strip().splitlines())
        lines.append("")
        lines.append(f"  sigma_noise = {result.sigma_noise:.6e} Jy/beam")
        lines.append("  Thresholds:")
        for m in args.multipliers:
            lines.append(
                f"    {m:>4.1f} x sigma_noise = {m * result.sigma_noise:.6e} Jy/beam"
            )

        out_png = args.out_dir / f"{target}_dirty_noise.png"
        plot_noise_validation(dirty_path, result, out_png)
        lines.append(f"  PNG saved: {out_png}")

        log_path = args.out_dir / f"{target}_dirty_noise.txt"
        log_path.write_text("\n".join(lines), encoding="utf-8")
        lines.append(f"  Log saved: {log_path}")

        _print_block(lines)
        summary.append((target, len(result.valid_patches), result.sigma_noise))

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'target':<30} {'valid_patches':>13} {'sigma_noise':>16}")
    for target, n_valid, sigma in summary:
        sigma_str = f"{sigma:.6e}" if sigma is not None else "FAILED"
        print(f"{target:<30} {n_valid:>13} {sigma_str:>16}")


def _print_block(lines: list[str]) -> None:
    print("\n".join(lines))
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Validate / characterize dirty-image noise estimation."
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Target names to process.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="RI-Benchmark project root.",
    )
    parser.add_argument(
        "--dirty-dir",
        type=Path,
        default=BASE_DIR / "results" / "dirty_noise_validation" / "casa_dirty",
        help="Directory containing pre-generated dirty FITS images.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=BASE_DIR / "results" / "dirty_noise_validation",
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--multipliers",
        nargs="+",
        type=float,
        default=[1.0, 1.5, 3.0],
        help="Threshold multipliers to report (validation mode only).",
    )
    parser.add_argument(
        "--characterize",
        action="store_true",
        help="Run 12-target characterization with CSV and scatter plot.",
    )
    args = parser.parse_args()

    if args.characterize:
        args.targets = args.targets or CHARACTERIZATION_TARGETS
        characterize_targets(args)
    else:
        args.targets = args.targets or VALIDATION_TARGETS
        validate_targets(args)


if __name__ == "__main__":
    main()
