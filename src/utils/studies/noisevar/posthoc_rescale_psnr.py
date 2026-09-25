"""Post-hoc rescaling experiment on noisevar deepinv reconstructions.

Loads the original deepinv reconstructions from the noisevar sweep (no re-run,
no rescaled FITS written — all rescaling is on the fly) and recomputes PSNR
against the unscaled GT (same criterion as src/compute_metrics.py: negatives
clipped, data_range = GT max - min).

Variants per cell:
    raw     : reconstruction as stored
    mm      : pure min-max to [0, 1]
    mmrob   : min-max with percentiles 0.1/99.9, clipped to [0, 1]
    clipmax : negatives to 0, divided by max

Usage:
    .envs/.benchmark/bin/python src/utils/studies/noisevar/posthoc_rescale_psnr.py --test
    .envs/.benchmark/bin/python src/utils/studies/noisevar/posthoc_rescale_psnr.py --full
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

BASE_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BASE_DIR))

from src.compute_metrics import clip_negatives, compute_ground_truth_metrics  # noqa: E402
from src.utils.noisevar_paths import noisevar_dirs, noisevar_gt_path  # noqa: E402

CONFIGS = ["l0p1_i50", "l1_i50"]
LEVELS = ["snr50", "snr200", "snr500"]
VARIANTS = ["raw", "mm", "mmrob", "clipmax"]

STATE_CSV = BASE_DIR / "results/noisevar/sweep/noisevar_state.csv"
OUT_TEST = BASE_DIR / "results/noisevar/posthoc_rescale_psnr_test.csv"
OUT_FULL = BASE_DIR / "results/noisevar/posthoc_rescale_psnr.csv"

CLASS_RE = re.compile(r"^disk_(s\d+(?:p\d+)?)_")


def load_fits(path):
    return fits.getdata(str(path)).squeeze().astype(np.float64)


def apply_variant(x, variant):
    if variant == "raw":
        return x
    if variant == "mm":
        lo, hi = x.min(), x.max()
        return np.zeros_like(x) if hi - lo == 0 else (x - lo) / (hi - lo)
    if variant == "mmrob":
        lo, hi = np.percentile(x, [0.1, 99.9])
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    if variant == "clipmax":
        x = np.clip(x, 0.0, None)
        return np.zeros_like(x) if x.max() == 0 else x / x.max()
    raise ValueError(variant)


def cell_paths(target, config):
    exp_root = Path(noisevar_dirs(str(BASE_DIR), target, "exp"))
    cell_dir = exp_root / target / f"{target}_deepinv" / f"{target}_deepinv_{config}"
    recon = cell_dir / f"{target}_deepinv_{config}.fits"
    return recon, cell_dir


def eval_cell(target, config):
    """One row per variant for a single (target, config) cell."""
    recon_path, _ = cell_paths(target, config)
    gt_path = noisevar_gt_path(str(BASE_DIR), target)
    if not recon_path.exists() or not gt_path:
        print(f"[warn] missing recon or GT for {target} {config}")
        return []
    x = load_fits(recon_path)
    gt = load_fits(gt_path)
    m = CLASS_RE.match(target)
    clase = m.group(1) if m else ""
    nivel = re.search(r"_snr(\d+)$", target).group(0).lstrip("_")
    rows = []
    for variant in VARIANTS:
        xv = apply_variant(x, variant)
        metrics = compute_ground_truth_metrics(clip_negatives(xv), gt)
        rows.append({
            "target": target, "clase": clase, "nivel": nivel,
            "config": config, "variante": variant, "psnr": metrics["psnr"],
        })
    return rows


def test_cells():
    """5 hand-picked cells: s0p5/s2/s4 x snr50/snr500, mixed."""
    picks = [
        ("disk_s0p5_r1_c0_x0_000_snr50", "disk_s0p5_r1_c0_x0_000"),
        ("disk_s2_r1_c0_x0_000_snr50", "disk_s2_r1_c0_x0_000"),
        ("disk_s4_r1_c0_x0_000_snr50", "disk_s4_r1_c0_x0_000"),
        ("disk_s0p5_r1_c0_x0_001_snr500", "disk_s0p5_r1_c0_x0_001"),
        ("disk_s2_r1_c0_x0_001_snr500", "disk_s2_r1_c0_x0_001"),
    ]
    return [(t, c) for t, _ in picks for c in CONFIGS]


def full_cells():
    """All completed deepinv cells (l0p1_i50, l1_i50) from the sweep state CSV."""
    state = pd.read_csv(STATE_CSV)
    mask = (
        (state["method"] == "deepinv")
        & (state["config_id"].isin(CONFIGS))
        & (state["status"] == "completed")
    )
    return list(state.loc[mask, ["target", "config_id"]].itertuples(index=False, name=None))


def summarize(df):
    pivot = df.groupby(["nivel", "config", "variante"])["psnr"].mean().unstack("variante")
    pivot = pivot.reindex(index=pd.MultiIndex.from_product(
        [LEVELS, CONFIGS], names=["nivel", "config"]))
    print("\n=== PSNR medio por nivel x config x variante ===")
    print(pivot.round(2).to_string())
    global_row = df.groupby("variante")["psnr"].mean().reindex(VARIANTS)
    print("\n=== Global (todas las celdas) ===")
    print(global_row.round(2).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test", action="store_true", help="5 celdas de prueba")
    group.add_argument("--full", action="store_true", help="todas las celdas completadas")
    args = parser.parse_args()

    cells = test_cells() if args.test else full_cells()
    out_csv = OUT_TEST if args.test else OUT_FULL
    print(f"Celdas a evaluar: {len(cells)}")

    rows = []
    for i, (target, config) in enumerate(cells, 1):
        rows.extend(eval_cell(target, config))
        if i % 200 == 0:
            print(f"  {i}/{len(cells)}")

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nCSV: {out_csv} ({len(df)} filas)")
    summarize(df)


if __name__ == "__main__":
    main()
