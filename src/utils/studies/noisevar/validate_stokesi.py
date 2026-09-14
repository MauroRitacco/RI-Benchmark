#!/usr/bin/env python3
"""P3 consistency check, final XX=YY convention: casa dirty vs deepinv
internal dirty on ONE noisevar cell (signal mode).

  --step casa     tclean niter=0 on the cell MS -> sigma_whole (nanstd of the
                  dirty image) and peak.  Run with .envs/.casa/bin/python.
  --step deepinv  exact replica of deepinverse.py:130-168 (RadioInterferometry,
                  nWimag = briggs*nW, x_hat = A_adjoint(nWimag*y)) on the cell
                  .mat -> sigma (nanstd) and peak.  Run with .envs/.deepinv.
  --step compare  reads results/noisevar/stokesi_check.json and applies the
                  checks: sigma_dinv/nW vs sigma_casa (+-5%) and the
                  scale-free peak/sigma ratio cross-check (+-5%).

State file: results/noisevar/stokesi_check.json (merged across steps).
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

BASE_DIR = Path(__file__).resolve().parents[4]
NOISEVAR = BASE_DIR / "data" / "noisevar"
STATE = BASE_DIR / "results" / "noisevar" / "stokesi_check.json"
TOL = 0.05


def cell_name(target, level):
    return f"{target}_snr{level}", NOISEVAR / "signal" / f"{target}_snr{level}"


def step_casa(target, level):
    from astropy.io import fits
    from casatasks import exportfits, tclean

    name, tdir = cell_name(target, level)
    ms = tdir / f"{name}.MS"
    pixel = float(loadmat(str(tdir / f"{name}.mat"))["nominal_pixelsize"].item())
    with tempfile.TemporaryDirectory(dir=str(STATE.parent)) as tmp:
        imagename = str(Path(tmp) / "dirty")
        tclean(
            vis=str(ms), imagename=imagename, imsize=64,
            cell=f"{pixel}arcsec", specmode="mfs", deconvolver="hogbom",
            gridder="standard", weighting="briggs", robust=2, stokes="I",
            datacolumn="data", niter=0, cyclefactor=0.0, savemodel="none",
        )
        if (Path(tmp) / "dirty.image").exists():
            (Path(tmp) / "dirty.image").rename(Path(tmp) / "dirty.dirty")
        exportfits(imagename=str(Path(tmp) / "dirty.dirty"),
                   fitsimage=str(Path(tmp) / "dirty_dirty.fits"), overwrite=True)
        shutil.copy(
            str(Path(tmp) / "dirty_dirty.fits"),
            str(STATE.parent / f"{name}_casa_dirty.fits"),
        )
        data = fits.getdata(str(Path(tmp) / "dirty_dirty.fits")).squeeze()
    out = {
        "sigma_casa": float(np.nanstd(data)),
        "peak_casa": float(np.nanmax(data)),
    }
    print(f"[casa] {name}: sigma_whole={out['sigma_casa']:.6e} "
          f"peak={out['peak_casa']:.6e}")
    return out


def step_deepinv(target, level):
    import torch
    from deepinv.physics import RadioInterferometry

    name, tdir = cell_name(target, level)
    mat = loadmat(str(tdir / f"{name}.mat"))
    y = torch.tensor(mat["y"].reshape(-1), dtype=torch.complex64)
    n_w = torch.tensor(mat["nW"].reshape(-1), dtype=torch.float32)
    briggs = torch.tensor(mat["weight"].reshape(-1), dtype=torch.float32)
    u = torch.tensor(mat["u"].reshape(-1), dtype=torch.float64)
    v = torch.tensor(mat["v"].reshape(-1), dtype=torch.float64)
    pixel = float(mat["nominal_pixelsize"].item())
    cell_rad = pixel * np.pi / (180.0 * 3600.0)
    u = (u * (2.0 * np.pi * cell_rad)).to(torch.float32)
    v = (v * (2.0 * np.pi * cell_rad)).to(torch.float32)
    uv = torch.stack([-v, u], dim=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nWimag = (briggs * n_w).view(1, 1, -1).to(device)
    y_w = (y.view(1, 1, -1).to(device) * nWimag)
    physics = RadioInterferometry(
        img_size=(64, 64), samples_loc=uv.permute((1, 0)).to(device),
        real_projection=True, device=device,
    )
    physics.setWeight(nWimag)
    with torch.no_grad():
        x_hat = physics.A_adjoint(y_w)
    img = x_hat.detach().cpu().numpy().squeeze()
    np.save(str(STATE.parent / f"{name}_dinv_dirty.npy"), img)
    out = {
        "sigma_dinv": float(np.nanstd(img)),
        "peak_dinv": float(np.nanmax(img)),
        "nW": float(n_w[0]),
    }
    print(f"[deepinv] {name}: sigma={out['sigma_dinv']:.6e} "
          f"peak={out['peak_dinv']:.6e} nW={out['nW']:.6e}")
    return out


def pixel_correlation(key):
    """Pearson r between the casa and deepinv dirty images (scale-free)."""
    from astropy.io import fits
    casa = fits.getdata(str(STATE.parent / f"{key}_casa_dirty.fits")).squeeze()
    dinv = np.load(str(STATE.parent / f"{key}_dinv_dirty.npy")).squeeze()
    a = casa.ravel().astype(float)
    b = dinv.ravel().astype(float)
    scale = float(np.dot(a, b) / np.dot(b, b))
    r = float(np.corrcoef(a, b)[0, 1])
    return r, scale


def step_compare(target, level):
    state = json.loads(STATE.read_text())
    key = f"{target}_snr{level}"
    s = state[key]
    sigma_dinv_scaled = s["sigma_dinv"] / s["nW"]
    r_sigma = sigma_dinv_scaled / s["sigma_casa"]
    snr_casa = s["peak_casa"] / s["sigma_casa"]
    snr_dinv = s["peak_dinv"] / s["sigma_dinv"]
    r_snr = snr_dinv / snr_casa
    r_pix, scale_pix = pixel_correlation(key)
    ok_sigma = abs(r_sigma - 1.0) <= TOL
    ok_snr = abs(r_snr - 1.0) <= TOL
    print(f"\n== {key} (convencion XX=YY, nW = 1/sigma_vis)")
    print(f"  casa    : sigma_whole={s['sigma_casa']:.6e}  peak={s['peak_casa']:.4e}  "
          f"peak/sigma={snr_casa:.2f}")
    print(f"  deepinv : sigma/nW   ={sigma_dinv_scaled:.6e}  (sigma={s['sigma_dinv']:.4e}, "
          f"nW={s['nW']:.4e})  peak/sigma={snr_dinv:.2f}")
    print(f"  check sigma: {r_sigma:.4f}  {'OK' if ok_sigma else 'FAIL'} (tol +-5%; "
          f"falta kappa tclean<->NUFFT)")
    print(f"  check peak/sigma: {r_snr:.4f}  {'OK' if ok_snr else 'FAIL'} (tol +-5%)")
    print(f"  correlacion pixel a pixel (rescalada): r={r_pix:.5f}  "
          f"escala a->b={scale_pix:.4e}")
    ok = ok_snr and r_pix >= 0.999
    s.update({"sigma_dinv_scaled": sigma_dinv_scaled, "ratio_sigma": r_sigma,
              "snr_casa": snr_casa, "snr_dinv": snr_dinv, "ratio_snr": r_snr,
              "corr_pix": r_pix, "scale_pix": scale_pix, "ok": ok})
    state[key] = s
    STATE.write_text(json.dumps(state, indent=2))
    print(f"\nWrote {STATE}")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=("casa", "deepinv", "compare"),
                        required=True)
    parser.add_argument("--target", default="disk_s2_r1_c0_x0_000")
    parser.add_argument("--level", type=int, default=50)
    args = parser.parse_args()

    STATE.parent.mkdir(parents=True, exist_ok=True)
    if args.step == "compare":
        sys.exit(step_compare(args.target, args.level))
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    key = f"{args.target}_snr{args.level}"
    state.setdefault(key, {}).update(
        step_casa(args.target, args.level)
        if args.step == "casa"
        else step_deepinv(args.target, args.level)
    )
    STATE.write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
