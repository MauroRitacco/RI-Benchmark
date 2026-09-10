#!/usr/bin/env python3
"""Analysis plots for the 360-target pure-noise validation.

Reads results/noise_validation/pure_noise_val360.csv and writes a single
2x2 figure to results/noise_snr/pure_noise_val360_plots.png:

  1. Histogram of sigma_measured (corner estimator, plane detrend) per class.
  2. sigma_measured vs sigma_theory per class, with the per-class linear
     fit (calibration factor) annotated in the legend.
  3. Histogram of sigma_measured / sigma_corners_nodetrend per class
     (detrending bias).
  4. Histogram of rms_corners_nodetrend / sigma_corners_nodetrend per class
     (rms~std check at patch level).

Also prints how many targets per class have sigma_measured outside
+/-3 sigma of their class mean, listing their names.
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "results" / "noise_validation" / "pure_noise_val360.csv"
PNG_PATH = BASE_DIR / "results" / "noise_snr" / "pure_noise_val360_plots.png"

CLASSES = ["s0p5", "s2", "s4"]
COLORS = {"s0p5": "tab:blue", "s2": "tab:orange", "s4": "tab:green"}
UJY = 1e6  # Jy/beam -> uJy/beam


def main():
    rows = list(csv.DictReader(open(CSV_PATH)))
    by_class = {c: [r for r in rows if r["class"] == c] for c in CLASSES}
    for c in CLASSES:
        n = len(by_class[c])
        if n != 120:
            print(f"[warn] clase {c}: {n} targets (esperados 120)")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel 1: histograma de sigma_measured por clase (uJy/beam).
    ax = axes[0, 0]
    for c in CLASSES:
        sig = np.array(
            [float(r["sigma_measured_jy_per_beam"]) for r in by_class[c]]
        ) * UJY
        ax.hist(
            sig, bins=25, alpha=0.55, color=COLORS[c],
            label=f"{c} (n={len(sig)})",
        )
    ax.set_xlabel(r"$\sigma_{medido}$ (µJy/beam)")
    ax.set_ylabel("N° de targets")
    ax.set_title("σ medido en esquinas (con detrend), por clase")
    ax.legend()

    # Panel 2: sigma_measured vs sigma_theory con ajuste lineal por clase.
    ax = axes[0, 1]
    for c in CLASSES:
        sth = np.array(
            [float(r["sigma_theory_jy_per_beam"]) for r in by_class[c]]
        ) * UJY
        sig = np.array(
            [float(r["sigma_measured_jy_per_beam"]) for r in by_class[c]]
        ) * UJY
        fit = np.polyfit(sth, sig, 1)
        ax.scatter(sth, sig, s=14, alpha=0.6, color=COLORS[c],
                   label=f"{c}: ajuste y={fit[0]:.3f}x{'+' if fit[1]>=0 else ''}{fit[1]:.1f}")
        xs = np.linspace(sth.min(), sth.max(), 100)
        ax.plot(xs, np.polyval(fit, xs), color=COLORS[c], lw=1.5)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="y=x")
    ax.set_xlabel(r"$\sigma_{teórico}$ (µJy/beam)")
    ax.set_ylabel(r"$\sigma_{medido}$ (µJy/beam)")
    ax.set_title("Calibración: σ medido vs teórico")
    ax.legend(fontsize=8)

    # Panel 3: ratio detrend/nodetrend por clase.
    ax = axes[1, 0]
    for c in CLASSES:
        sig = np.array(
            [float(r["sigma_measured_jy_per_beam"]) for r in by_class[c]]
        )
        signd = np.array(
            [float(r["sigma_corners_nodetrend_jy_per_beam"]) for r in by_class[c]]
        )
        ratio = sig / signd
        ax.hist(
            ratio, bins=25, alpha=0.55, color=COLORS[c],
            label=f"{c}: media={ratio.mean():.3f}",
        )
    ax.axvline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel(r"$\sigma_{detrend} / \sigma_{sin\,detrend}$")
    ax.set_ylabel("N° de targets")
    ax.set_title("Sesgo del detrending de plano")
    ax.legend()

    # Panel 4: rms/std a nivel parche (sin detrend) por clase.
    ax = axes[1, 1]
    for c in CLASSES:
        rmsnd = np.array(
            [float(r["rms_corners_nodetrend_jy_per_beam"]) for r in by_class[c]]
        )
        signd = np.array(
            [float(r["sigma_corners_nodetrend_jy_per_beam"]) for r in by_class[c]]
        )
        ratio = rmsnd / signd
        ax.hist(
            ratio, bins=25, alpha=0.55, color=COLORS[c],
            label=f"{c}: media={ratio.mean():.3f}",
        )
    ax.axvline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel(r"rms/$\sigma$ en esquinas (sin detrend)")
    ax.set_ylabel("N° de targets")
    ax.set_title("Chequeo rms≈std a nivel parche")
    ax.legend()

    fig.suptitle("Validación de ruido puro — 360 targets de validación")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=150)
    print(f"PNG: {PNG_PATH}")

    # Outliers: sigma_measured fuera de +/-3 sigma de la media de su clase.
    print("\nOutliers de sigma_measured (fuera de +/-3σ de la media de clase):")
    total = 0
    for c in CLASSES:
        sig = np.array(
            [float(r["sigma_measured_jy_per_beam"]) for r in by_class[c]]
        )
        mu, sd = sig.mean(), sig.std()
        mask = np.abs(sig - mu) > 3 * sd
        total += int(mask.sum())
        print(f"  {c}: {int(mask.sum())} (media={mu*UJY:.1f} µJy/beam, σ={sd*UJY:.1f})")
        for r, m in zip(by_class[c], mask):
            if m:
                v = float(r["sigma_measured_jy_per_beam"])
                print(f"    - {r['target']}: {v*UJY:.1f} µJy/beam")
    print(f"  TOTAL: {total}")


if __name__ == "__main__":
    sys.exit(main())
