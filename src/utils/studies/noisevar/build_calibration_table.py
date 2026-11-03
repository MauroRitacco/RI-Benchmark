#!/usr/bin/env python3
# Noisevar step 1: sigma_vis per (target, level) so the dirty image of the
# fixed GT reaches peak SNR 50/200/500. Reads the unit-noise measurements in
# results/noise_validation/bright/pure_noise_val360.csv; writes
# results/noisevar/calibration_table.csv (1080 rows).
import csv
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[4]
CSV_PATH = BASE_DIR / "results" / "noise_validation" / "bright" / "pure_noise_val360.csv"
OUT_DIR = BASE_DIR / "results" / "noisevar"
OUT_CSV = OUT_DIR / "calibration_table.csv"

LEVELS = (50, 200, 500)
HEADER = [
    "target", "clase", "nivel",
    "dirty_peak_jy_per_beam", "sigma_whole_jy_per_beam",
    "sigma_imagen_objetivo_jy_per_beam", "sigma_vis",
]


def main():
    if not CSV_PATH.exists():
        sys.exit(f"Input CSV not found: {CSV_PATH}")

    rows = []
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            target = row["target"]
            cls = row["class"]
            dirty_peak = float(row["dirty_peak_jy_per_beam"])
            sigma_whole = float(row["sigma_whole_jy_per_beam"])
            for L in LEVELS:
                sigma_img = dirty_peak / L
                sigma_vis = sigma_img / sigma_whole
                rows.append([
                    target, cls, L,
                    f"{dirty_peak:.6e}", f"{sigma_whole:.6e}",
                    f"{sigma_img:.6e}", f"{sigma_vis:.6e}",
                ])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")

    # Report sigma_vis range per class and level.
    by_cls_lvl = defaultdict(list)
    for r in rows:
        by_cls_lvl[(r[1], int(r[2]))].append(float(r[6]))
    print(f"\n{'clase':6} {'nivel':>6} {'n':>4} {'sigma_vis min':>14} "
          f"{'media':>12} {'max':>14}")
    for (cls, L) in sorted(by_cls_lvl):
        vals = by_cls_lvl[(cls, L)]
        n = len(vals)
        print(f"{cls:6} {L:>6} {n:>4} {min(vals):>14.4e} "
              f"{sum(vals) / n:>12.4e} {max(vals):>14.4e}")


if __name__ == "__main__":
    main()
