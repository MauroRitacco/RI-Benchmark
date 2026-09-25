#!/usr/bin/env python3
"""Hyper-parameter sweep driver for CASA tclean, WSClean and DeepInverse.

Resumable: completed outputs are skipped on re-runs.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.io import loadmat

# Make project imports available
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from src.compute_metrics import clip_negatives, compute_ground_truth_metrics, extract_runtime
from src.utils.wsclean_dirty_noise import get_wsclean_sigma_noise
from src.utils.noises_paths import snr_level_dir
from src.utils.noisevar_paths import noisevar_level_dir

# {method: [(config_id, params_dict), ...]}
SWEEP_GRIDS = None


# Sweep grids are read from a text file (one line per swept parameter,
# '#' starts a comment):
#   method,param,value1,value2,...
# e.g.  casa,gain,0.1,0.2
import itertools

CASA_BASE = {
    "imsize": 64,
    "specmode": "mfs",
    "deconvolver": "hogbom",
    "gridder": "standard",
    "weighting": "briggs",
    "robust": 2,
    "stokes": "I",
    "datacolumn": "data",
    "niter": 1000000000,
    "cyclefactor": 0.0,
    "savemodel": "none",
}

WSCLEAN_BASE = {
    "size": 64,
    "weight": ["briggs", 2],
    "mgain": 1.0,  # single deep major cycle, matching CASA cyclefactor=0.0 behaviour
    "niter": 1000000000,
    "nmiter": 0,  # unlimited major cycles, matching CASA's convergence to threshold
    "j": 1,
    "no-update-model-required": True,
    "data-column": "DATA",
    "pol": "I",
}

DEEPINV_BASE = {
    "img_size": [64, 64],
    "real_projection": True,
    "early_stop": True,
    "thres_conv": 1e-5,
    "stepsize_factor": 1.5,
    "prior": "wavelet",
}

# Fixed R2D2 configuration taken from data/simulated/disk_1_5/disk_1_5.yaml.
R2D2_BASE = {
    "imager_config": "config/imaging/R2D2_small.yaml",
    "architecture": "unet",
    "prune": True,
    "sigma_res_tol": 0.0001,
    "super_resolution": 1,
    "meas_op_on_gpu": True,
    "im_dim_x": 64,
    "im_dim_y": 64,
    "weight_robustness": 2,
    "weight_type": "briggs",
    "num_iter": 5,
    "series": "R2D2",
    "layers": 1,
    "num_chans": 32,
    "ckpt_path": "ckpt/R2D2_small_ALMA3/",
    "ckpt_realisations": 1,
}

# Short tags used in config_ids.  The order of params in the sweep file
# determines the order of tags in the config_id, so it is stable.
PARAM_TAGS = {
    "gain": "g",
    "threshold": "t",
    "lambda_factor": "l",
    "max_iter": "i",
}


def _coerce_value(v):
    """Convert a string value from the sweep file to int/float/str."""
    v = v.strip()
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def parse_sweep_config(path):
    """Parse sweep file into {method: {param: [values]}} preserving declaration order."""
    grids = {}
    with open(path, "r") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            method, param = parts[0], parts[1]
            values = [_coerce_value(v) for v in parts[2:]]
            if method not in grids:
                grids[method] = {}
            grids[method][param] = values
    return grids


def _config_id_from_params(params):
    """Build a short config id from a {param: value} dict, in insertion order.

    'auto:' thresholds use a 'ts' tag so they never collide with legacy 't' ids.
    The 'abs3s' token (absolute 3-sigma threshold from the pure-noise
    characterization) uses its own 'a3s' tag.
    """
    parts = []
    for param, value in params.items():
        if param == "threshold" and value == ABS3S_TOKEN:
            parts.append("a3s")
            continue
        tag = PARAM_TAGS.get(param, param[0])
        if param == "threshold" and isinstance(value, str) and value.startswith("auto:"):
            tag = "ts"
        parts.append(f"{tag}{_fmt(value)}")
    return "_".join(parts)


def build_sweep_grids(sweep_path):
    """Return {method: [(config_id, params), ...]} over the Cartesian product of params.

    A single 'config' param is a fixed config id (e.g. R2D2, no swept params).
    """
    grids = parse_sweep_config(sweep_path)
    result = {}
    for method, params in grids.items():
        keys = list(params.keys())
        values = [params[k] for k in keys]
        combos = []
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            if keys == ["config"] and len(combo) == 1:
                cid = str(combo[0])
                combos.append((cid, {}))
            else:
                cid = _config_id_from_params(param_dict)
                combos.append((cid, param_dict))
        result[method] = combos
    return result


def build_casa_config(params, imagename):
    """Return a full tclean kwargs dict from swept params."""
    cfg = dict(CASA_BASE)
    cfg.update(params)
    if "threshold" in cfg:
        th = cfg["threshold"]
        # Auto thresholds are resolved inside casa.py from the dirty image.
        if not (isinstance(th, str) and th.startswith("auto:")):
            cfg["threshold"] = f"{th}Jy"
    cfg["imagename"] = str(imagename)
    return cfg


def build_wsclean_args(target, pixel_size, prefix, params):
    """Build the WSClean CLI argument list from swept params."""
    ms = target_dir(BASE_DIR, target) / f"{target}.MS"
    args = ["wsclean"]
    args += ["-size", "64", "64"]
    args += ["-scale", f"{pixel_size}asec"]
    args += ["-weight", "briggs", "2"]
    # Use a single w-layer so WSClean behaves like a 2D gridder, matching CASA's
    # standard gridder and DeepInverse's NUFFT.
    args += ["-gridder", "wstacking"]
    args += ["-wstack-nwlayers", "1"]
    args += ["-mgain", "0.8"]
    args += ["-niter", "1000000000"]
    args += ["-nmiter", "0"]  # unlimited major cycles: converge to absolute threshold like CASA
    args += ["-j", "1"]
    args += ["-no-update-model-required"]
    if "gain" in params:
        args += ["-gain", str(params["gain"])]
    if "threshold" in params:
        args += ["-abs-threshold", str(params["threshold"])]
    args += ["-data-column", "DATA"]
    args += ["-pol", "I"]
    args += ["-name", str(prefix)]
    args += [str(ms)]
    return args


def build_deepinv_config(params, out_dir, opnorm=None, skip_warmup=True):
    """Return a full DeepInverse config dict from swept params."""
    cfg = dict(DEEPINV_BASE)
    cfg.update(params)
    cfg["out_dir"] = str(out_dir) + "/"
    if opnorm is not None:
        cfg["opnorm"] = opnorm
    cfg["skip_warmup"] = skip_warmup
    return cfg


def build_r2d2_config(params):
    """Return a full R2D2 config dict from the fixed base plus any overrides."""
    cfg = dict(R2D2_BASE)
    cfg.update(params)
    return cfg


# Special modes: "test" (final evaluation, single config per method, 1 worker),
# "debil" (deprecated alias of the faint-SNR study, kept for the old dataset)
# and "noises" (faint-SNR study over the 3-level noises dataset: inputs at
# data/simulated/noises/<L>s/<target>/, outputs under experiments/simulated/
# noises/ and results/simulated/noises/). Subprocesses see the mode in
# RI_BENCH_MODE (RI_BENCH_TEST=1 is also set for test, for backwards
# compatibility with older wrappers).
MODE = ""


def data_root():
    """Directory holding the input target folders (MS/.mat/FITS)."""
    if MODE == "noisevar":
        return BASE_DIR / "data" / "noisevar" / "noisevar"
    if MODE:
        return BASE_DIR / "data" / "simulated" / MODE
    return BASE_DIR / "data" / "simulated"


def exp_root():
    """Root directory for reconstruction outputs."""
    if MODE == "noises":
        return BASE_DIR / "experiments" / "simulated" / "noises"
    if MODE == "noisevar":
        return BASE_DIR / "experiments" / "noisevar"
    if MODE:
        return BASE_DIR / "experiments" / MODE
    return BASE_DIR / "experiments" / "simulated"


def results_root():
    """Root directory for sweep state files and reports."""
    if MODE == "noises":
        return BASE_DIR / "results" / "simulated" / "noises"
    if MODE == "noisevar":
        return BASE_DIR / "results" / "noisevar" / "sweep"
    if MODE:
        return BASE_DIR / "results" / MODE
    return BASE_DIR / "results" / "sweep"


def target_dir(base_dir, target):
    if MODE == "noises":
        lvl = snr_level_dir(target)
        if lvl:
            return data_root() / lvl / target
    if MODE == "noisevar":
        lvl = noisevar_level_dir(target)
        if lvl:
            return data_root() / lvl / target
    return data_root() / target


def state_csv_name():
    """Sweep state CSV name for the current mode."""
    return f"{MODE}_state.csv" if MODE else "sweep_state.csv"


def results_report_name():
    """Evaluation report name for the current mode."""
    return f"{MODE}_results.txt" if MODE else "sweep_results.txt"


def exp_target_dir(base_dir, target):
    if MODE == "noises":
        lvl = snr_level_dir(target)
        if lvl:
            return exp_root() / lvl / target
    if MODE == "noisevar":
        lvl = noisevar_level_dir(target)
        if lvl:
            return exp_root() / lvl / target
    return exp_root() / target


def opnorm_cache_path(target):
    """Path to the cached operator norm for a target (next to its .mat file)."""
    return target_dir(BASE_DIR, target) / f"{target}_opnorm.json"


def load_cached_opnorm(target):
    """Return the cached operator norm for target, or None if not available."""
    path = opnorm_cache_path(target)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return float(json.load(f)["opnorm"])
    except Exception as e:
        print(f"[prep] {target}: could not read opnorm cache: {e}")
        return None


def sweep_dir(base_dir):
    d = results_root()
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_pixel_size(target):
    mat_path = target_dir(BASE_DIR, target) / f"{target}.mat"
    return float(loadmat(mat_path)["nominal_pixelsize"].item())


def _is_auto_threshold(threshold):
    """Return True if the threshold value is an auto multiplier like 'auto:3'."""
    return isinstance(threshold, str) and threshold.startswith("auto:")


# 'abs3s' threshold token: absolute 3-sigma threshold taken from the pure-noise
# characterization (results/noisevar/calibration_table.csv) instead of the
# per-image sigma estimated during cleaning. Noisevar mode only.
ABS3S_TOKEN = "abs3s"
_ABS3S_TABLE_CACHE = {}


def abs3s_threshold(target):
    """Resolve the 'abs3s' token to an absolute Jy threshold for one cell.

    thr = 3 * sigma_imagen_objetivo, where sigma_imagen_objetivo =
    sigma_vis * sigma_whole from results/noisevar/calibration_table.csv
    (image-domain noise in Jy/beam). Returns None if not found.
    """
    if not _ABS3S_TABLE_CACHE:
        path = BASE_DIR / "results" / "noisevar" / "calibration_table.csv"
        with open(path) as f:
            for row in csv.DictReader(f):
                _ABS3S_TABLE_CACHE[(row["target"], row["nivel"])] = float(
                    row["sigma_imagen_objetivo_jy_per_beam"])
    level = noisevar_level_dir(target).replace("snr", "")
    base_target = target[: -len(noisevar_level_dir(target)) - 1] if level else target
    sigma_pure = _ABS3S_TABLE_CACHE.get((base_target, level))
    return None if sigma_pure is None else 3.0 * sigma_pure


def _auto_multiplier(threshold):
    """Extract the numeric multiplier from an 'auto:X' string."""
    return float(threshold.split(":", 1)[1])


def _find_gt_path(base_dir, target):
    """Return the ground-truth FITS path if it exists, else None."""
    if MODE:
        gt = target_dir(base_dir, target) / f"{target}.fits"
        return str(gt) if gt.exists() else None
    for category in ("simulated", "archival"):
        gt = base_dir / "data" / category / target / f"{target}.fits"
        if gt.exists():
            return str(gt)
    return None


def gt_fits_path(target):
    """Ground-truth FITS for metrics (never rescaled).

    noisevar: the GT is fixed and lives OUTSIDE the cell; read the absolute
    gt_fits reference from the cell's .json (written by the generator). The
    original unscaled FITS is used as-is. Other modes: <target>.fits in the
    target dir.
    """
    if MODE == "noisevar":
        from src.utils.noisevar_paths import noisevar_gt_path
        return noisevar_gt_path(str(BASE_DIR), target)
    return _find_gt_path(BASE_DIR, target)


def load_state(state_csv):
    """Return dict mapping (target, method, config_id) -> row dict.

    Rows still 'running' from an interrupted run become 'interrupted' so they are retried.
    """
    state_rows = {}
    if not state_csv.exists():
        return state_rows
    with open(state_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["target"], row["method"], row["config_id"])
            if row["status"] == "running":
                row["status"] = "interrupted"
                row["time"] = ""
            state_rows[key] = row
    return state_rows


# Columns persisted in the sweep state CSV.
STATE_HEADER = ["target", "method", "config_id", "status", "time", "psnr", "ssim", "mse", "t"]


def _write_state_locked(state_csv, state_rows):
    """Write the full state dict to CSV. Caller must hold the lock."""
    state_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(state_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATE_HEADER)
        writer.writeheader()
        for row in state_rows.values():
            writer.writerow({k: row.get(k, "") for k in STATE_HEADER})


def save_state(state_csv, lock, state_rows):
    """Write the full state dict to CSV, keeping one row per (target,method,config_id)."""
    with lock:
        _write_state_locked(state_csv, state_rows)


def update_state_row(state_csv, lock, state_rows, row):
    """Insert or update a state row and persist the CSV.

    The insert and the write are atomic under the lock: otherwise a worker
    adding a new key while another worker writes the CSV triggers
    "dictionary changed size during iteration" and kills the worker.
    """
    key = (row["target"], row["method"], row["config_id"])
    with lock:
        state_rows[key] = row
        _write_state_locked(state_csv, state_rows)


def count_job_states(state_rows):
    """Return counts of completed, failed and running jobs."""
    completed = sum(1 for r in state_rows.values() if r.get("status") == "completed")
    failed = sum(1 for r in state_rows.values() if r.get("status") == "failed")
    running = sum(1 for r in state_rows.values() if r.get("status") == "running")
    return completed, failed, running


def progress_reporter(state_rows, state_lock, stop_event, total_jobs, interval=60):
    """Print a progress summary to /dev/tty every interval seconds (fallback: stderr)."""
    while not stop_event.is_set():
        stop_event.wait(interval)
        if stop_event.is_set():
            break
        with state_lock:
            completed, failed, running = count_job_states(state_rows)
        msg = f"[progress] {completed}/{total_jobs} completed | {running} running | {failed} failed"
        try:
            with open("/dev/tty", "w") as tty:
                tty.write(msg + "\n")
                tty.flush()
        except Exception:
            print(msg, file=sys.stderr)


def evaluate_state(state_csv, output_path):
    """Analyze sweep_state.csv and write a text report with per-cell averages.

    Only rows with status='completed' are included.
    """
    if not state_csv.exists():
        print(f"State file not found: {state_csv}")
        sys.exit(1)

    rows = []
    with open(state_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    warmup = [r for r in rows if r.get("config_id") == "warmup"]
    completed = [r for r in rows if r.get("status") == "completed" and r.get("config_id") != "warmup"]
    failed = [r for r in rows if r.get("status") == "failed" and r.get("config_id") != "warmup"]
    running = [r for r in rows if r.get("status") in ("running", "interrupted") and r.get("config_id") != "warmup"]

    for r in completed:
        for k in ("psnr", "ssim", "mse", "time", "t"):
            try:
                r[k] = float(r.get(k, ""))
            except (ValueError, TypeError):
                r[k] = None

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def _std(vals):
        vals = [v for v in vals if v is not None]
        return statistics.stdev(vals) if len(vals) > 1 else 0.0

    def _fov_class(target):
        parts = (target or "").split("_")
        return parts[1] if len(parts) > 1 else ""

    def build_stats(completed_rows):
        grp = {}
        for r in completed_rows:
            key = (r["method"], r["config_id"])
            grp.setdefault(key, []).append(r)
        out = []
        for (method, config_id), g in sorted(grp.items()):
            out.append({
                "method": method,
                "config_id": config_id,
                "n": len(g),
                "psnr_mean": _mean([r["psnr"] for r in g]),
                "psnr_std": _std([r["psnr"] for r in g]),
                "ssim_mean": _mean([r["ssim"] for r in g]),
                "ssim_std": _std([r["ssim"] for r in g]),
                "psnr_fov": {
                    cls: _mean([r["psnr"] for r in g if _fov_class(r["target"]) == cls])
                    for cls in ("s0p5", "s2", "s4")
                },
                "mse_mean": _mean([r["mse"] for r in g]),
                "mse_std": _std([r["mse"] for r in g]),
                "t_mean": _mean([r["t"] for r in g]),
                "t_std": _std([r["t"] for r in g]),
                "time_mean": _mean([r["time"] for r in g]),
                "time_std": _std([r["time"] for r in g]),
            })
        return out

    stats = build_stats(completed)

    methods = sorted(set(s["method"] for s in stats))

    def _best(metric, reverse):
        valid = [s for s in stats if s[f"{metric}_mean"] is not None]
        if not valid:
            return None
        return sorted(valid, key=lambda s: s[f"{metric}_mean"], reverse=reverse)[0]

    global_best = {
        "psnr": _best("psnr", True),
        "ssim": _best("ssim", True),
        "mse": _best("mse", False),
    }

    top3_by_method = {}
    for method in methods:
        method_stats = sorted(
            [s for s in stats if s["method"] == method and s["psnr_mean"] is not None],
            key=lambda s: s["psnr_mean"],
            reverse=True,
        )
        top3_by_method[method] = method_stats[:3]

    lines = []
    lines.append("=" * 80)
    lines.append("RI-Benchmark Sweep Evaluation")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"State file: {state_csv}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Global summary")
    lines.append("-" * 80)
    lines.append(f"Total rows in state file: {len(rows)}")
    lines.append(f"Completed jobs: {len(completed)}")
    lines.append(f"Failed jobs: {len(failed)}")
    lines.append(f"Running/interrupted jobs: {len(running)}")
    lines.append(f"Warmup rows: {len(warmup)}")
    lines.append("")

    lines.append("Per-method summary")
    lines.append("-" * 80)
    for method in methods:
        method_completed = [r for r in completed if r["method"] == method]
        method_failed = [r for r in failed if r["method"] == method]
        best = top3_by_method[method][0] if top3_by_method[method] else None
        lines.append(f"Method: {method}")
        lines.append(f"  Completed jobs: {len(method_completed)}")
        lines.append(f"  Failed jobs: {len(method_failed)}")
        n_configs = len(SWEEP_GRIDS[method]) if (SWEEP_GRIDS and method in SWEEP_GRIDS) else 0
        lines.append(f"  Configurations tried: {n_configs}")
        if SWEEP_GRIDS and method in SWEEP_GRIDS and SWEEP_GRIDS[method]:
            param_names = list(SWEEP_GRIDS[method][0][1].keys())
            for pname in param_names:
                vals = sorted({params[pname] for _, params in SWEEP_GRIDS[method]})
                lines.append(f"    {pname}={vals}")
        if best:
            psnr_var = best['psnr_std'] ** 2 if best['psnr_std'] is not None else 0.0
            lines.append(
                f"  Best PSNR: {best['config_id']} = {best['psnr_mean']:.2f} dB "
                f"(std={best['psnr_std']:.2f}, var={psnr_var:.2f}, "
                f"SSIM={best['ssim_mean']:.4f}, MSE={best['mse_mean']:.6f}, "
                f"n={best['n']}, t={best['t_mean']:.2f}s, time={best['time_mean']:.2f}s)"
            )
        if method_completed:
            for metric in ("psnr", "ssim", "mse", "t"):
                vals = [r[metric] for r in method_completed if r[metric] is not None]
                if vals:
                    lo, hi = min(vals), max(vals)
                    lines.append(f"  {metric.upper()} range: [{lo:.4f}, {hi:.4f}]")
        lines.append("")

    lines.append("Global rankings by average")
    lines.append("-" * 80)
    if global_best["psnr"]:
        b = global_best["psnr"]
        lines.append(f"Best PSNR (higher better): {b['method']} {b['config_id']} = {b['psnr_mean']:.4f} (n={b['n']}, t={b['t_mean']:.2f}s, time={b['time_mean']:.2f}s)")
    else:
        lines.append("Best PSNR (higher better): N/A")
    if global_best["ssim"]:
        b = global_best["ssim"]
        lines.append(f"Best SSIM (higher better): {b['method']} {b['config_id']} = {b['ssim_mean']:.4f} (n={b['n']}, t={b['t_mean']:.2f}s, time={b['time_mean']:.2f}s)")
    else:
        lines.append("Best SSIM (higher better): N/A")
    if global_best["mse"]:
        b = global_best["mse"]
        lines.append(f"Best MSE (lower better):  {b['method']} {b['config_id']} = {b['mse_mean']:.4f} (n={b['n']}, t={b['t_mean']:.2f}s, time={b['time_mean']:.2f}s)")
    else:
        lines.append("Best MSE (lower better): N/A")
    lines.append("")

    lines.append("Top 3 configs per method by average PSNR")
    lines.append("-" * 80)
    for method in methods:
        lines.append(f"Method: {method}")
        if top3_by_method[method]:
            for i, s in enumerate(top3_by_method[method], 1):
                lines.append(f"  {i}. {s['config_id']}: PSNR={s['psnr_mean']:.2f}, SSIM={s['ssim_mean']:.4f}, MSE={s['mse_mean']:.4f} (n={s['n']}, t={s['t_mean']:.2f}s, time={s['time_mean']:.2f}s)")
        else:
            lines.append("  No completed configs for this method.")
        lines.append("")

    # In noises/noisevar mode, show mean PSNR per FOV class (s0p5/s2/s4) next to psnr.
    FOV_CLASSES = ("s0p5", "s2", "s4")
    show_fov = MODE in ("noises", "noisevar")
    fov_hdr = "".join(f" {('psnr_' + c):>10}" for c in FOV_CLASSES) if show_fov else ""
    table_hdr = (
        f"{'method':<12} {'config_id':<20} {'n':>5} {'psnr':>8}"
        f"{fov_hdr} {'ssim':>8} {'mse':>8} {'t(s)':>8} {'time(s)':>8}"
    )

    def fmt_row(s):
        fov_cells = ""
        if show_fov:
            cells = []
            for c in FOV_CLASSES:
                v = s["psnr_fov"][c]
                cells.append(f"{v:>10.2f}" if v is not None else f"{'-':>10}")
            fov_cells = " " + " ".join(cells)
        return (
            f"{s['method']:<12} {s['config_id']:<20} {s['n']:>5} "
            f"{s['psnr_mean']:>8.2f}{fov_cells} {s['ssim_mean']:>8.4f} "
            f"{s['mse_mean']:>8.4f} {s['t_mean']:>8.2f} {s['time_mean']:>8.2f}"
        )

    lines.append("All configurations (averages)")
    lines.append("-" * 80)
    lines.append(table_hdr)
    for s in sorted(stats, key=lambda x: (x["method"], x["config_id"])):
        lines.append(fmt_row(s))
    lines.append("")

    if MODE in ("noises", "noisevar"):
        # Per-SNR-level tables: same layout as the global table above, one
        # table per level (level dir derived from '_snr<L>' suffix).
        if MODE == "noises":
            levels = ("50s", "200s", "500s")
            lvl_of = snr_level_dir
        else:
            levels = ("snr50", "snr200", "snr500")
            lvl_of = noisevar_level_dir
        for lvl in levels:
            lvl_rows = [r for r in completed if lvl_of(r["target"]) == lvl]
            lvl_stats = build_stats(lvl_rows)
            lines.append(f"All configurations (averages) — nivel {lvl}")
            lines.append("-" * 80)
            lines.append(table_hdr)
            if lvl_stats:
                for s in sorted(lvl_stats, key=lambda x: (x["method"], x["config_id"])):
                    lines.append(fmt_row(s))
            else:
                lines.append("(no completed jobs at this level yet)")
            lines.append("")

    text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nReport written to: {output_path}")


def evaluate_job(target, method, config_id):
    """Compute PSNR/SSIM/MSE and imaging time for a finished sweep cell."""
    if MODE:
        gt_path = Path(gt_fits_path(target) or "")
        cell_dir = exp_target_dir(BASE_DIR, target) / f"{target}_{method}" / f"{target}_{method}_{config_id}"
    elif os.path.exists(str(BASE_DIR / "data" / "simulated" / target)):
        gt_path = BASE_DIR / "data" / "simulated" / target / f"{target}.fits"
        cell_dir = BASE_DIR / "experiments" / "simulated" / target / f"{target}_{method}" / f"{target}_{method}_{config_id}"
    elif os.path.exists(str(BASE_DIR / "data" / "archival" / target)):
        gt_path = BASE_DIR / "data" / "archival" / target / f"{target}.fits"
        cell_dir = BASE_DIR / "experiments" / "archival" / target / f"{target}_{method}" / f"{target}_{method}_{config_id}"
    else:
        return {"psnr": "", "ssim": "", "mse": "", "imaging_time": ""}
    recon_path = cell_dir / f"{target}_{method}_{config_id}.fits"
    # CASA/WSClean name logs after config_id; deepinv and r2d2 wrappers use their
    # own log names (early r2d2 cells only have the wrapper log).
    if method == "deepinv":
        log_path = cell_dir / f"{target}_deepinverse.log"
    elif method == "r2d2":
        log_path = cell_dir / f"{target}_r2d2_{config_id}.log"
        if not log_path.exists():
            log_path = cell_dir / f"{target}_r2d2.log"
    else:
        log_path = cell_dir / f"{target}_{method}_{config_id}.log"

    if not recon_path.exists() or not gt_path.exists():
        return {"psnr": "", "ssim": "", "mse": "", "imaging_time": ""}

    try:
        gt_data = fits.getdata(str(gt_path)).squeeze().astype(np.float64)
        recon_data = fits.getdata(str(recon_path)).squeeze().astype(np.float64)
        recon_data = clip_negatives(recon_data)

        m = compute_ground_truth_metrics(recon_data, gt_data)
        metrics = {
            "psnr": f"{m['psnr']:.2f}",
            "ssim": f"{m['ssim']:.4f}",
            "mse": f"{m['mse']:.4f}",
        }
    except Exception as e:
        print(f"[warn] Could not compute metrics for {target} {method} {config_id}: {e}")
        metrics = {"psnr": "", "ssim": "", "mse": ""}

    # extract_runtime expects the method tag "deepinverse", not "deepinv".
    runtime_method = "deepinverse" if method == "deepinv" else method
    t = extract_runtime(str(log_path), runtime_method)
    if np.isnan(t):
        metrics["t"] = ""
    else:
        metrics["t"] = f"{t:.2f}"

    return metrics


def backfill_completed_metrics(state_rows):
    """Compute metrics for completed rows that don't have them yet.

    The wall-clock 'time' column is left untouched: it must not be overwritten
    by the imaging time 't' extracted from the method log.
    """
    for (target, method, config_id), row in state_rows.items():
        if row.get("status") != "completed":
            continue
        needs_metrics = not (row.get("psnr") and row.get("ssim") and row.get("mse") and row.get("t"))
        if not needs_metrics:
            continue
        metrics = evaluate_job(target, method, config_id)
        row.update(metrics)


def init_csv(path, header):
    """Create an empty CSV with the given header if it does not exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def read_or_create_gt_index(index_csv, target):
    """Return the GT total flux for target, creating the index row if needed."""
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    if index_csv.exists():
        with open(index_csv, "r", newline="") as f:
            for row in csv.DictReader(f):
                if row["target"] == target:
                    return float(row["gt_total_flux"])

    gt_path = gt_fits_path(target)
    if gt_path is None:
        raise FileNotFoundError(f"GT FITS not found for {target}")
    gt = fits.getdata(gt_path).squeeze().astype(np.float64)
    total = gt.sum()

    init_csv(index_csv, ["target", "gt_total_flux"])
    with open(index_csv, "a", newline="") as f:
        csv.writer(f).writerow([target, f"{total:.4f}"])
    return total


def prepare_target(target):
    """Ensure GT index entry and DeepInverse weights exist for a target.

    Weights live inside the target's .mat file; the old
    experiments/<target>/<target>_weights/ dir is removed if present.
    """
    index_csv = sweep_dir(BASE_DIR) / "gt_flux_index.csv"
    gt_total = read_or_create_gt_index(index_csv, target)
    print(f"[prep] {target}: GT total flux = {gt_total:.4f}")

    mat_path = target_dir(BASE_DIR, target) / f"{target}.mat"
    weights_ok = False
    if mat_path.exists():
        try:
            mat_data = loadmat(mat_path)
            weights_ok = "weight" in mat_data and mat_data["weight"].size > 0
        except Exception:
            weights_ok = False

    old_weights_dir = exp_target_dir(BASE_DIR, target) / f"{target}_weights"
    if old_weights_dir.exists():
        try:
            shutil.rmtree(old_weights_dir)
            print(f"[prep] {target}: removed obsolete weights dir {old_weights_dir}")
        except Exception as e:
            print(f"[prep] {target}: could not remove old weights dir: {e}")

    if not weights_ok:
        print(f"[prep] {target}: generating weights ...")
        cmd = [str(BASE_DIR / "scripts" / "generate_weights.sh"), target, "2"]
        result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[prep] WARNING: generate_weights failed for {target}:\n{result.stderr}")
        else:
            print(f"[prep] {target}: weights done")
    else:
        print(f"[prep] {target}: weights already cached")



def _fmt(v):
    """Format a value for file/folder names: 'auto:X' -> 'X', '.' -> 'p'."""
    s = f"{v}"
    if isinstance(v, str) and s.startswith("auto:"):
        s = s.split(":", 1)[1]
    return s.replace(".", "p")


def is_done(target, method, config_id):
    """Return True if a valid, non-empty FITS output already exists."""
    cell_dir = exp_target_dir(BASE_DIR, target) / f"{target}_{method}" / f"{target}_{method}_{config_id}"
    final_path = cell_dir / f"{target}_{method}_{config_id}.fits"
    if not final_path.exists():
        return False
    try:
        hdr = fits.getheader(str(final_path))
        return hdr.get("NAXIS", 0) > 0 and final_path.stat().st_size > 2880
    except Exception:
        return False


def clean_cell_dir(target, method, config_id):
    """Remove any partial/stale outputs from a previous failed/interrupted run."""
    cell_dir = exp_target_dir(BASE_DIR, target) / f"{target}_{method}" / f"{target}_{method}_{config_id}"
    if not cell_dir.exists():
        return
    try:
        for f in cell_dir.iterdir():
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()
        print(f"[clean] {target} {method} {config_id} cell dir reset")
    except Exception as e:
        print(f"[warn] Could not clean {cell_dir}: {e}")


def run_casa(target, config_id, config, dry_run=False):
    out = exp_target_dir(BASE_DIR, target)
    cell_dir = out / f"{target}_casa" / f"{target}_casa_{config_id}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    imagename = cell_dir / f"{target}_casa_{config_id}"
    final_path = cell_dir / f"{target}_casa_{config_id}.fits"

    cfg = dict(config)
    cfg["imagename"] = str(imagename)

    cmd = [
        str(BASE_DIR / ".envs" / ".casa" / "bin" / "python"),
        str(BASE_DIR / "src" / "algorithms" / "casa.py"),
        "--target", target,
        "--config", json.dumps(cfg),
        "--nval", "1",
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return 0.0, True

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=out, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0 or not final_path.exists():
        print(f"[error] CASA {target} {config_id} failed:\n{result.stderr}")
        return elapsed, False

    return elapsed, True


def run_wsclean(target, pixel_size, config_id, params, dry_run=False):
    out = exp_target_dir(BASE_DIR, target)
    cell_dir = out / f"{target}_wsclean" / f"{target}_wsclean_{config_id}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    prefix = cell_dir / f"{target}_wsclean_{config_id}"
    source_path = prefix.with_name(f"{prefix.name}-image.fits")
    final_path = cell_dir / f"{target}_wsclean_{config_id}.fits"
    log_path = cell_dir / f"{target}_wsclean_{config_id}.log"

    params = dict(params)

    # Resolve auto thresholds from a WSClean dirty pass (cached per target).
    if _is_auto_threshold(params.get("threshold")):
        multiplier = _auto_multiplier(params["threshold"])
        if dry_run:
            params["threshold"] = "AUTO"  # placeholder for dry-run print
        else:
            ms = target_dir(BASE_DIR, target) / f"{target}.MS"
            gt_path = gt_fits_path(target)
            dirty_prefix = out / f"{target}_wsclean_dirty_pass"
            cache_path = out / ".wsclean_dirty_noise.json"

    cmd = build_wsclean_args(target, pixel_size, str(prefix), params)

    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return 0.0, True

    t0 = time.perf_counter()
    with open(log_path, "w") as logf:
        def _log(msg, *args):
            logf.write((msg % args) + "\n")
            logf.flush()

        if _is_auto_threshold(params.get("threshold")) or (
            "threshold" in params and str(params["threshold"]).startswith("AUTO")
        ):
            # Only real auto thresholds reach here (dry-run uses the "AUTO" placeholder).
            noise_result = get_wsclean_sigma_noise(
                ms_path=ms,
                size=64,
                pixel_size=pixel_size,
                weight=["briggs", 2],
                data_column="DATA",
                dirty_prefix=dirty_prefix,
                gt_path=gt_path,
                cache_path=cache_path,
                cwd=out,
                log_func=_log,
            )
            computed_threshold = multiplier * noise_result.sigma_noise
            params["threshold"] = computed_threshold
            _log("[noise] WSClean auto threshold resolved from dirty image")
            _log("[noise] HAS_EXT=%s", noise_result.has_point_source)
            if noise_result.source_px is not None:
                _log(
                    "[noise] point source pixel=(row=%.2f, col=%.2f)",
                    noise_result.source_px[0], noise_result.source_px[1]
                )
            for p in noise_result.valid_patches:
                _log(
                    "[noise] patch %s VALID: mean=%.6e std=%.6e rms=%.6e n=%d",
                    p.name, p.mean, p.std, p.rms, p.n
                )
            for d in noise_result.discarded_patches:
                extra = ""
                if "mean" in d and "std" in d:
                    extra = f" mean={d['mean']:.6e} std={d['std']:.6e} rms={d['rms']:.6e}"
                elif "std" in d:
                    extra = f" std={d['std']:.6e}"
                _log("[noise] patch %s DISCARDED: %s%s", d["patch"], d["reason"], extra)
            _log(
                "[noise] sigma_noise=%.6e (mean std of %d valid patches)",
                noise_result.sigma_noise, len(noise_result.valid_patches)
            )
            _log("[noise] multiplier=%.2f -> threshold=%.6e", multiplier, computed_threshold)

            cmd = build_wsclean_args(target, pixel_size, str(prefix), params)

        if "threshold" in params and not str(params["threshold"]).startswith("AUTO"):
            _log("[noise] absolute threshold=%.6e Jy (resolved from sweep config)",
                 float(params["threshold"]))

        result = subprocess.run(cmd, cwd=out, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0 or not source_path.exists():
        print(f"[error] WSClean {target} {config_id} failed (exit {result.returncode})")
        return elapsed, False

    # Convert Jy/beam -> Jy/pixel, then copy to the final name.
    from src.utils.transforms import convert_jybeam_to_jypixel
    convert_jybeam_to_jypixel(str(source_path))
    shutil.copy2(source_path, final_path)
    return elapsed, True


def run_deepinverse(target, config_id, config, dry_run=False):
    out = exp_target_dir(BASE_DIR, target)
    cell_dir = out / f"{target}_deepinv" / f"{target}_deepinv_{config_id}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    final_path = cell_dir / f"{target}_deepinv_{config_id}.fits"
    source_path = cell_dir / f"{target}_deepinverse.fits"

    cfg = dict(config)
    cfg["out_dir"] = str(cell_dir) + "/"

    cmd = [
        str(BASE_DIR / ".envs" / ".deepinv" / "bin" / "python"),
        str(BASE_DIR / "src" / "algorithms" / "deepinverse.py"),
        "--target", target,
        "--config", json.dumps(cfg),
        "--nval", "1",
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"

    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return 0.0, True

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=out, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0 or not source_path.exists():
        print(f"[error] DeepInv {target} {config_id} failed:\n{result.stderr}")
        return elapsed, False

    shutil.copy2(source_path, final_path)
    return elapsed, True


def run_r2d2(target, config_id, config, dry_run=False):
    """Run R2D2 via scripts/run_r2d2.sh, then rename the generic <target>_r2d2.*
    outputs to the sweep's <target>_r2d2_<config_id>.* naming so evaluate_job
    can locate them."""
    out = exp_target_dir(BASE_DIR, target)
    cell_dir = out / f"{target}_r2d2" / f"{target}_r2d2_{config_id}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    source_fits = cell_dir / f"{target}_r2d2.fits"
    source_log = cell_dir / f"{target}_r2d2.log"
    final_fits = cell_dir / f"{target}_r2d2_{config_id}.fits"
    final_log = cell_dir / f"{target}_r2d2_{config_id}.log"

    cmd = [
        str(BASE_DIR / "scripts" / "run_r2d2.sh"),
        "--target", target,
        "--output-dir", str(cell_dir.absolute()),
    ]
    if config:
        cmd += ["--config", json.dumps(config)]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"

    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return 0.0, True

    t0 = time.perf_counter()
    # Capture stdout/stderr and write the log after the wrapper returns.  We cannot
    # keep final_log open during the call because run_r2d2.sh removes and recreates
    # its output directory, which would delete the still-open log file from the tree.
    result = subprocess.run(cmd, cwd=out, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - t0

    with open(final_log, "w") as logf:
        logf.write(result.stdout)

    if result.returncode != 0 or not source_fits.exists():
        print(f"[error] R2D2 {target} {config_id} failed (exit {result.returncode})")
        return elapsed, False

    shutil.copy2(source_fits, final_fits)
    return elapsed, True


def run_deepinv_warmup(target, state_csv, state_lock, state_rows):
    """Run one DeepInverse warm-up job to compute/cache the operator norm.

    Also serves as the global CUDA warm-up; recorded in sweep_state.csv as
    config_id='warmup' with empty metric columns.
    """
    out = exp_target_dir(BASE_DIR, target)
    cell_dir = out / f"{target}_deepinv" / f"{target}_deepinv_warmup"
    cell_dir.mkdir(parents=True, exist_ok=True)

    source_path = cell_dir / f"{target}_deepinverse.fits"
    final_path = cell_dir / f"{target}_deepinv_warmup.fits"

    cfg = build_deepinv_config({}, cell_dir, opnorm=None, skip_warmup=False)

    cmd = [
        str(BASE_DIR / ".envs" / ".deepinv" / "bin" / "python"),
        str(BASE_DIR / "src" / "algorithms" / "deepinverse.py"),
        "--target", target,
        "--config", json.dumps(cfg),
        "--nval", "1",
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"

    print(f"[prep] {target}: warming up DeepInv (computing opnorm) ...")
    update_state_row(state_csv, state_lock, state_rows, {
        "target": target,
        "method": "deepinv",
        "config_id": "warmup",
        "status": "running",
        "time": "",
    })
    clean_cell_dir(target, "deepinv", "warmup")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=out, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0 or not source_path.exists():
        print(f"[error] DeepInv warm-up failed for {target}:\n{result.stderr}")
        update_state_row(state_csv, state_lock, state_rows, {
            "target": target,
            "method": "deepinv",
            "config_id": "warmup",
            "status": "failed",
            "time": f"{elapsed:.2f}",
        })
        return None

    shutil.copy2(source_path, final_path)
    opnorm = load_cached_opnorm(target)
    status = "completed"
    update_state_row(state_csv, state_lock, state_rows, {
        "target": target,
        "method": "deepinv",
        "config_id": "warmup",
        "status": status,
        "time": f"{elapsed:.2f}",
    })
    opnorm_str = f"{opnorm:.6f}" if opnorm is not None else "unknown"
    print(f"[prep] {target}: warm-up completed in {elapsed:.2f}s (opnorm = {opnorm_str})")
    return opnorm


def run_cpu_jobs_for_target(target, pixel_size, state_csv, state_lock, state_rows, dry_run):
    """Run all CASA and WSClean configurations for a single target."""
    jobs = []
    for method in ("casa", "wsclean"):
        if method not in SWEEP_GRIDS:
            continue
        for cid, params in SWEEP_GRIDS[method]:
            jobs.append((method, cid, params))

    for method, cid, params in jobs:
        if is_done(target, method, cid):
            print(f"[skip] {target} {method} {cid} already done")
            continue

        params = dict(params)
        if params.get("threshold") == ABS3S_TOKEN:
            thr = abs3s_threshold(target)
            if thr is None:
                print(f"[error] {target} {method} {cid}: abs3s threshold not found "
                      f"in calibration table")
                update_state_row(state_csv, state_lock, state_rows, {
                    "target": target, "method": method, "config_id": cid,
                    "status": "failed", "time": "0",
                })
                continue
            params["threshold"] = thr
            print(f"[noise] {target} {method} {cid}: abs3s threshold = {thr:.6e} Jy")

        print(f"[run]  {target} {method} {cid}")
        update_state_row(state_csv, state_lock, state_rows, {
            "target": target,
            "method": method,
            "config_id": cid,
            "status": "running",
            "time": "",
        })
        if not dry_run:
            clean_cell_dir(target, method, cid)

        if method == "casa":
            out = exp_target_dir(BASE_DIR, target)
            cell_dir = out / f"{target}_casa" / f"{target}_casa_{cid}"
            config = build_casa_config(params, cell_dir / f"{target}_casa_{cid}")
            elapsed, ok = run_casa(target, cid, config, dry_run=dry_run)
        else:
            elapsed, ok = run_wsclean(target, pixel_size, cid, params, dry_run=dry_run)

        status = "completed" if ok else "failed"
        row = {
            "target": target,
            "method": method,
            "config_id": cid,
            "status": status,
            "time": f"{elapsed:.2f}",
        }
        if status == "completed":
            row.update(evaluate_job(target, method, cid))
        update_state_row(state_csv, state_lock, state_rows, row)
        print(f"[{status}] {target} {method} {cid} in {elapsed:.2f}s")


def run_gpu_jobs_for_target(target, state_csv, state_lock, state_rows, dry_run):
    """Run GPU-bound configurations for a single target (R2D2 + DeepInverse).

    The DeepInverse operator norm comes from a warm-up run right before the
    target's configs if not already cached (cache kept on disk for resumed sweeps).
    """
    if "r2d2" in SWEEP_GRIDS:
        for cid, params in SWEEP_GRIDS["r2d2"]:
            if is_done(target, "r2d2", cid):
                print(f"[skip] {target} r2d2 {cid} already done")
                continue

            print(f"[run]  {target} r2d2 {cid}")
            update_state_row(state_csv, state_lock, state_rows, {
                "target": target,
                "method": "r2d2",
                "config_id": cid,
                "status": "running",
                "time": "",
            })
            if not dry_run:
                clean_cell_dir(target, "r2d2", cid)

            config = build_r2d2_config(params)
            elapsed, ok = run_r2d2(target, cid, config, dry_run=dry_run)
            status = "completed" if ok else "failed"
            row = {
                "target": target,
                "method": "r2d2",
                "config_id": cid,
                "status": status,
                "time": f"{elapsed:.2f}",
            }
            if status == "completed":
                row.update(evaluate_job(target, "r2d2", cid))
            update_state_row(state_csv, state_lock, state_rows, row)
            print(f"[{status}] {target} r2d2 {cid} in {row['time']}s")

    if "deepinv" not in SWEEP_GRIDS:
        return

    opnorm = load_cached_opnorm(target)
    if opnorm is None and not dry_run:
        opnorm = run_deepinv_warmup(target, state_csv, state_lock, state_rows)

    for cid, params in SWEEP_GRIDS["deepinv"]:
        if is_done(target, "deepinv", cid):
            print(f"[skip] {target} deepinv {cid} already done")
            continue

        print(f"[run]  {target} deepinv {cid}")
        update_state_row(state_csv, state_lock, state_rows, {
            "target": target,
            "method": "deepinv",
            "config_id": cid,
            "status": "running",
            "time": "",
        })
        if not dry_run:
            clean_cell_dir(target, "deepinv", cid)

        out = exp_target_dir(BASE_DIR, target)
        cell_dir = out / f"{target}_deepinv" / f"{target}_deepinv_{cid}"
        config = build_deepinv_config(params, cell_dir, opnorm=opnorm, skip_warmup=True)
        elapsed, ok = run_deepinverse(target, cid, config, dry_run=dry_run)
        status = "completed" if ok else "failed"
        row = {
            "target": target,
            "method": "deepinv",
            "config_id": cid,
            "status": status,
            "time": f"{elapsed:.2f}",
        }
        if status == "completed":
            row.update(evaluate_job(target, "deepinv", cid))
        update_state_row(state_csv, state_lock, state_rows, row)
        print(f"[{status}] {target} deepinv {cid} in {row['time']}s")



def cpu_worker(target_queue_cpu, cpu_target_locks, state_csv, state_lock, state_rows, dry_run):
    while True:
        target = target_queue_cpu.get()
        if target is None:
            target_queue_cpu.task_done()
            break
        try:
            pixel_size = read_pixel_size(target)
            with cpu_target_locks[target]:
                run_cpu_jobs_for_target(target, pixel_size, state_csv, state_lock, state_rows, dry_run)
        except Exception as e:
            print(f"[error] CPU worker crashed on {target}: {e}")
        finally:
            target_queue_cpu.task_done()


def gpu_worker(target_queue_gpu, gpu_target_locks, state_csv, state_lock, state_rows, dry_run):
    while True:
        target = target_queue_gpu.get()
        if target is None:
            target_queue_gpu.task_done()
            break
        try:
            with gpu_target_locks[target]:
                run_gpu_jobs_for_target(target, state_csv, state_lock, state_rows, dry_run)
        except Exception as e:
            print(f"[error] GPU worker crashed on {target}: {e}")
        finally:
            target_queue_gpu.task_done()


def main():
    parser = argparse.ArgumentParser(description="Hyper-parameter sweep for RI reconstruction methods.")
    parser.add_argument("--targets", default=None,
                        help="Text file with one target name per line. Default: sweep_targets.txt, or test_targets.txt with --test.")
    parser.add_argument("--sweep-config", default=None,
                        help="Text file with sweep grids (method,param,value1,... per line). Default: sweep_config.txt, or test_config.txt with --test.")
    parser.add_argument("--test", action="store_true",
                        help="Final-evaluation mode: read targets from data/simulated/test/, write to "
                             "experiments/test/ and results/test/. Exactly one config per method is required; "
                             "workers are forced to 1.")
    parser.add_argument("--noises", action="store_true",
                        help="Faint-SNR study mode over the 3-level noises dataset: read targets from "
                             "data/simulated/noises/<L>s/<target>/ (level dir derived from the '_snr<L>' "
                             "name suffix), write to experiments/simulated/noises/ and "
                             "results/simulated/noises/. The sweep grid is NOT restricted to a single "
                             "config per method.")
    parser.add_argument("--debil", action="store_true",
                        help="DEPRECATED alias of the old faint-SNR study (data/simulated/debil/, "
                             "experiments/debil/, results/debil/). Use --noises for the current study; "
                             "--debil is kept only to rerun the old dataset.")
    parser.add_argument("--noisevar", action="store_true",
                        help="GT-fixed + variable-noise study mode: read targets from "
                             "data/noisevar/noisevar/<snr<L>>/<target>/ (level dir derived "
                             "from the '_snr<L>' name suffix), GT from the cell .json "
                             "gt_fits reference (never rescaled), write to "
                             "experiments/noisevar/ and results/noisevar/sweep/.")
    parser.add_argument("--cpu-workers", type=int, default=1,
                        help="Number of CPU workers for CASA + WSClean targets.")
    parser.add_argument("--gpu-workers", type=int, default=1,
                        help="Number of GPU workers for DeepInverse targets.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them.")
    parser.add_argument("--no-prepare", action="store_true",
                        help="Skip the per-target preparation step (weights/GT index).")
    parser.add_argument("--evaluate", action="store_true",
                        help="Analyze sweep_state.csv and write a summary report without running any reconstruction.")
    args = parser.parse_args()

    global MODE
    modes = [args.test, args.noises, args.debil, args.noisevar]
    if sum(1 for m in modes if m) > 1:
        print("ERROR: --test, --noises, --debil and --noisevar are mutually exclusive.")
        sys.exit(1)
    if args.test:
        MODE = "test"
        os.environ["RI_BENCH_MODE"] = "test"
        os.environ["RI_BENCH_TEST"] = "1"
        if args.cpu_workers != 1 or args.gpu_workers != 1:
            print("[test] forcing --cpu-workers 1 and --gpu-workers 1")
        args.cpu_workers = 1
        args.gpu_workers = 1
    elif args.noises:
        MODE = "noises"
        os.environ["RI_BENCH_MODE"] = "noises"
    elif args.noisevar:
        MODE = "noisevar"
        os.environ["RI_BENCH_MODE"] = "noisevar"
    elif args.debil:
        print("WARNING: --debil is DEPRECATED (old faint-SNR study over data/simulated/debil/); "
              "use --noises for the current noises study.")
        MODE = "debil"
        os.environ["RI_BENCH_MODE"] = "debil"

    if args.targets is None:
        args.targets = str(BASE_DIR / "scripts" / (f"{MODE}_targets.txt" if MODE else "sweep_targets.txt"))
    if args.sweep_config is None:
        args.sweep_config = str(BASE_DIR / "scripts" / (f"{MODE}_config.txt" if MODE else "sweep_config.txt"))

    sweep_config_file = Path(args.sweep_config)
    if not sweep_config_file.exists():
        print(f"Sweep config file not found: {sweep_config_file}")
        sys.exit(1)

    # Load sweep grids into the global read-only structure.
    global SWEEP_GRIDS
    SWEEP_GRIDS = build_sweep_grids(sweep_config_file)
    print("Loaded sweep methods:", list(SWEEP_GRIDS.keys()))

    if MODE == "test":
        # Guardrail: the test set is evaluated once with a single winning config
        # per method, not swept. Abort if any method expands to more than one cell.
        for method, combos in SWEEP_GRIDS.items():
            if len(combos) != 1:
                print(f"[test] ERROR: method '{method}' has {len(combos)} configs in test mode; exactly 1 is required.")
                sys.exit(1)

    if args.evaluate:
        state_csv = sweep_dir(BASE_DIR) / state_csv_name()
        output_path = sweep_dir(BASE_DIR) / results_report_name()
        evaluate_state(state_csv, output_path)
        return

    targets_file = Path(args.targets)
    if not targets_file.exists():
        print(f"Targets file not found: {targets_file}")
        sys.exit(1)

    targets = [line.strip() for line in open(targets_file) if line.strip() and not line.startswith("#")]
    if not targets:
        print("No targets to process.")
        sys.exit(0)

    print(f"Loaded {len(targets)} targets: {targets}")

    sweep_root = sweep_dir(BASE_DIR)
    state_csv = sweep_root / state_csv_name()
    state_lock = threading.Lock()
    state_rows = load_state(state_csv)
    if not args.dry_run:
        backfill_completed_metrics(state_rows)
        save_state(state_csv, state_lock, state_rows)  # persist cleaning of stale 'running' rows + metrics backfill

    # Per-target locks: CPU and GPU phases never overlap, but two workers of
    # the same pool may still pick the same target.
    cpu_target_locks = {t: threading.Lock() for t in targets}
    gpu_target_locks = {t: threading.Lock() for t in targets}

    # GT flux index and DeepInverse weights (skipped on dry-run).
    if not args.dry_run and not args.no_prepare:
        for target in targets:
            prepare_target(target)

    total_cpu_jobs = 0
    total_gpu_jobs = 0
    for target in targets:
        for method in ("casa", "wsclean"):
            if method in SWEEP_GRIDS:
                for cid, _ in SWEEP_GRIDS[method]:
                    if not is_done(target, method, cid):
                        total_cpu_jobs += 1
        for method in ("deepinv", "r2d2"):
            if method in SWEEP_GRIDS:
                for cid, _ in SWEEP_GRIDS[method]:
                    if not is_done(target, method, cid):
                        total_gpu_jobs += 1
    print(f"Scheduled jobs: {total_cpu_jobs} CPU, {total_gpu_jobs} GPU")

    if args.dry_run:
        print("\n=== DRY RUN: printing one representative command per method ===")
        pixel_size = read_pixel_size(targets[0])
        if "casa" in SWEEP_GRIDS:
            cid, params = SWEEP_GRIDS["casa"][0]
            cell_dir = exp_target_dir(BASE_DIR, targets[0]) / f"{targets[0]}_casa" / f"{targets[0]}_casa_{cid}"
            print("CASA:", run_casa(targets[0], cid, build_casa_config(params, cell_dir / f"{targets[0]}_casa_{cid}"), dry_run=True))
        if "wsclean" in SWEEP_GRIDS:
            cid, params = SWEEP_GRIDS["wsclean"][0]
            print("WSClean:", run_wsclean(targets[0], pixel_size, cid, params, dry_run=True))
        if "r2d2" in SWEEP_GRIDS:
            cid, params = SWEEP_GRIDS["r2d2"][0]
            print("R2D2:", run_r2d2(targets[0], cid, build_r2d2_config(params), dry_run=True))
        if "deepinv" in SWEEP_GRIDS:
            cid, params = SWEEP_GRIDS["deepinv"][0]
            cell_dir = exp_target_dir(BASE_DIR, targets[0]) / f"{targets[0]}_deepinv" / f"{targets[0]}_deepinv_{cid}"
            print("DeepInv:", run_deepinverse(targets[0], cid, build_deepinv_config(params, cell_dir), dry_run=True))
        print("=== END DRY RUN ===\n")
        return

    from queue import Queue
    cpu_queue = Queue()
    gpu_queue = Queue()
    for target in targets:
        cpu_queue.put(target)
        gpu_queue.put(target)

    for _ in range(args.cpu_workers):
        cpu_queue.put(None)
    for _ in range(args.gpu_workers):
        gpu_queue.put(None)

    t_start = time.perf_counter()

    # Progress reporter: prints to /dev/tty every 60s to keep the stdout log clean.
    total_jobs = total_cpu_jobs + total_gpu_jobs
    progress_stop = threading.Event()
    progress_thread = None
    if total_jobs > 0 and not args.dry_run:
        progress_thread = threading.Thread(
            target=progress_reporter,
            args=(state_rows, state_lock, progress_stop, total_jobs, 60),
            daemon=True,
        )
        progress_thread.start()

    # Phase 1: CPU-bound jobs (CASA + WSClean) first, so GPU-bound DeepInverse
    # jobs do not contend for CPU/memory/NUFFT resources.
    if total_cpu_jobs > 0 and args.cpu_workers > 0:
        print(f"Phase 1/2: {total_cpu_jobs} CPU jobs with {args.cpu_workers} worker(s) ...")
        with ThreadPoolExecutor(max_workers=args.cpu_workers) as executor:
            futures = [
                executor.submit(cpu_worker, cpu_queue, cpu_target_locks, state_csv, state_lock, state_rows, False)
                for _ in range(args.cpu_workers)
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[error] CPU worker failed: {e}")
        print("Phase 1/2: CPU jobs finished.")

    if total_gpu_jobs > 0 and args.gpu_workers > 0:
        print(f"Phase 2/2: {total_gpu_jobs} GPU jobs with {args.gpu_workers} worker(s) ...")
        with ThreadPoolExecutor(max_workers=args.gpu_workers) as executor:
            futures = [
                executor.submit(gpu_worker, gpu_queue, gpu_target_locks, state_csv, state_lock, state_rows, False)
                for _ in range(args.gpu_workers)
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[error] GPU worker failed: {e}")
        print("Phase 2/2: GPU jobs finished.")

    if progress_thread is not None:
        progress_stop.set()
        progress_thread.join(timeout=5)

    elapsed = time.perf_counter() - t_start
    print(f"\nSweep finished in {elapsed:.2f}s")
    print(f"State file: {state_csv}")


if __name__ == "__main__":
    main()
