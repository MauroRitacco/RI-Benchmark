# Verifica las 1080 celdas noisevar en data/noisevar/noisevar.
# Convencion: XX=YY ("Stokes I efectivo"), nW = 1/sigma_vis.
# Modo signal (default, dataset del sweep): y = y_unit + ruido. Chequeos por
# celda: archivos presentes, campos del .mat, gt_fits valido, nW*sigma_vis~=1,
# residual (y - y_unit)*nW con rms~=1 (y_unit recomputado del GT original con
# la misma clean_visibilities del generador — nunca de faint/).
# Modo noise (celdas de ruido puro): rms(|y|)*nW~=1.
import argparse
import csv
import json
import sys
import numpy as np
from pathlib import Path
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_noisevar_ms import clean_visibilities  # noqa: E402

REQUIRED = ["y", "nW", "u", "v", "w", "frequency", "nominal_pixelsize", "weight"]


def check_cell(cell: Path, sigma_vis: float, target: str, mode: str, rtol_files=0.02):
    name = cell.name
    ms = list(cell.glob("*.MS"))
    mats = list(cell.glob("*.mat"))
    jsons = list(cell.glob("*.json"))
    if len(ms) != 1 or len(mats) != 1 or len(jsons) != 1:
        return name, False, f"archivos: MS={len(ms)} mat={len(mats)} json={len(jsons)}"
    meta = json.loads(jsons[0].read_text())
    if mode == "signal":
        gt = Path(meta.get("gt_fits", ""))
        if not gt.is_file():
            return name, False, f"gt_fits invalido: {meta.get('gt_fits')}"
    m = loadmat(mats[0])
    missing = [k for k in REQUIRED if k not in m]
    if missing:
        return name, False, f"faltan campos {missing}"
    y = np.asarray(m["y"]).ravel()
    nW = float(np.mean(np.asarray(m["nW"]).ravel()))
    if not np.isfinite(nW) or nW <= 0:
        return name, False, f"nW invalido {nW}"
    r_nw = nW * sigma_vis
    if abs(r_nw - 1) > 1e-4:
        return name, False, f"nW*sv={r_nw:.6f}"
    if mode == "noise":
        r = float(np.sqrt(np.mean(np.abs(y) ** 2))) * nW
    else:
        _, y_unit = clean_visibilities(target, Path("data/simulated") / target / f"{target}.mat")
        r = float(np.sqrt(np.mean(np.abs(y - np.asarray(y_unit).ravel()) ** 2))) * nW
    ok = abs(r - 1) < rtol_files
    msg = f"nW*sv={r_nw:.6f} residual*nW rms={r:.4f}"
    return name, ok, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/noisevar/noisevar")
    ap.add_argument("--table", default="results/noisevar/calibration_table.csv")
    ap.add_argument("--out", default="results/noisevar/verify_1080.json")
    ap.add_argument("--mode", choices=("signal", "noise"), default="signal")
    args = ap.parse_args()

    with open(args.table) as f:
        sv = {(r["target"], int(r["nivel"])): float(r["sigma_vis"]) for r in csv.DictReader(f)}

    cells = []
    for base in [Path(args.root), *sorted(Path(args.root).glob("snr*"))]:
        if base.is_dir():
            cells.extend(sorted(d for d in base.iterdir()
                                if d.is_dir() and not d.name.startswith("snr")))
    cells = sorted(cells)
    fails, passes = [], 0
    rows = []
    for cell in cells:
        parts = cell.name.rsplit("_snr", 1)
        if len(parts) != 2:
            fails.append((cell.name, "nombre sin _snr"))
            continue
        target, nivel = parts[0], int(parts[1])
        if (target, nivel) not in sv:
            fails.append((cell.name, "no esta en calibration_table"))
            continue
        name, ok, msg = check_cell(cell, sv[(target, nivel)], target, args.mode)
        rows.append({"cell": name, "ok": ok, "detail": msg})
        if ok:
            passes += 1
        else:
            fails.append((name, msg))

    report = {"total": len(cells), "pass": passes, "fail": len(fails), "failures": fails}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"total={len(cells)} pass={passes} fail={len(fails)}")
    for name, msg in fails[:20]:
        print(f"FAIL {name}: {msg}")


if __name__ == "__main__":
    main()
