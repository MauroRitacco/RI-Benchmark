#!/usr/bin/env python3
"""Hyper-parameter sweep driver for CASA tclean, WSClean and DeepInverse.

Runs the full Cartesian product of configured grids for every target listed in
a text file, reuses the existing Python wrappers and the WSClean binary, and
renames the produced FITS files so that each configuration can be identified.

The driver is resumable: completed outputs are skipped on re-runs.
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

# Make project imports available (e.g. src.utils.transforms)
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

# Reuse metric helpers from the benchmark harness
from src.compute_metrics import clip_negatives, compute_ground_truth_metrics, extract_runtime
from src.utils.wsclean_dirty_noise import get_wsclean_sigma_noise

# Parsed sweep configuration: {method: [(config_id, params_dict), ...]}
SWEEP_GRIDS = None


# -----------------------------------------------------------------------------
# Sweep grids -- read from a simple text file instead of hardcoded constants.
#
# Format (one line per swept parameter, no header, '#' starts a comment):
#   method,param,value1,value2,...
#
# Example:
#   casa,gain,0.1,0.2
#   casa,threshold,0.01,0.05,0.1,0.2,0.5
#   wsclean,gain,0.1,0.2
#   wsclean,threshold,0.01,0.05,0.1,0.2,0.5
#   deepinv,lambda_factor,5e-4,1e-3,5e-3
#   deepinv,max_iter,100,200
# -----------------------------------------------------------------------------
import itertools

# Base configurations that are kept fixed for each method.
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
    """Build a short config id from a {param: value} dict.

    Params are processed in insertion order (Python 3.7+ dicts preserve order).
    """
    parts = []
    for param, value in params.items():
        tag = PARAM_TAGS.get(param, param[0])
        parts.append(f"{tag}{_fmt(value)}")
    return "_".join(parts)


def build_sweep_grids(sweep_path):
    """Return a dict of generators: {method: [(config_id, params), ...]}.

    The product is taken over the declared params for each method.
    A special single-value parameter named 'config' is interpreted as a fixed
    config id (used for methods such as R2D2 that do not sweep hyperparameters).
    """
    grids = parse_sweep_config(sweep_path)
    result = {}
    for method, params in grids.items():
        keys = list(params.keys())
        values = [params[k] for k in keys]
        combos = []
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            # Fixed-config shortcut: method,config,<id> yields config_id=<id> and no params.
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


# -----------------------------------------------------------------------------
# Paths and utilities
# -----------------------------------------------------------------------------
# Test mode (--test): inputs are read from data/simulated/test/, outputs go to
# experiments/test/ and results/test/. Controlled by the TEST_MODE global set
# in main(); subprocesses are told via the RI_BENCH_TEST environment variable.
TEST_MODE = False


def data_root():
    """Directory holding the input target folders (MS/.mat/FITS)."""
    if TEST_MODE:
        return BASE_DIR / "data" / "simulated" / "test"
    return BASE_DIR / "data" / "simulated"


def exp_root():
    """Root directory for reconstruction outputs."""
    if TEST_MODE:
        return BASE_DIR / "experiments" / "test"
    return BASE_DIR / "experiments" / "simulated"


def results_root():
    """Root directory for sweep state files and reports."""
    if TEST_MODE:
        return BASE_DIR / "results" / "test"
    return BASE_DIR / "results" / "sweep"


def target_dir(base_dir, target):
    return data_root() / target


def exp_target_dir(base_dir, target):
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


def _auto_multiplier(threshold):
    """Extract the numeric multiplier from an 'auto:X' string."""
    return float(threshold.split(":", 1)[1])


def _find_gt_path(base_dir, target):
    """Return the ground-truth FITS path if it exists, else None."""
    if TEST_MODE:
        gt = data_root() / target / f"{target}.fits"
        return str(gt) if gt.exists() else None
    for category in ("simulated", "archival"):
        gt = base_dir / "data" / category / target / f"{target}.fits"
        if gt.exists():
            return str(gt)
    return None


# -----------------------------------------------------------------------------
# State / index bookkeeping
# -----------------------------------------------------------------------------
def load_state(state_csv):
    """Return dict mapping (target, method, config_id) -> row dict.

    Any row still marked 'running' from a previous interrupted run is converted to
    'interrupted' so that it is retried on the next pass.
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


def save_state(state_csv, lock, state_rows):
    """Write the full state dict to CSV, keeping one row per (target,method,config_id)."""
    with lock:
        state_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(state_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=STATE_HEADER)
            writer.writeheader()
            for row in state_rows.values():
                writer.writerow({k: row.get(k, "") for k in STATE_HEADER})


def update_state_row(state_csv, lock, state_rows, row):
    """Insert or update a state row and persist the CSV."""
    key = (row["target"], row["method"], row["config_id"])
    state_rows[key] = row
    save_state(state_csv, lock, state_rows)


def count_job_states(state_rows):
    """Return counts of completed, failed and running jobs."""
    completed = sum(1 for r in state_rows.values() if r.get("status") == "completed")
    failed = sum(1 for r in state_rows.values() if r.get("status") == "failed")
    running = sum(1 for r in state_rows.values() if r.get("status") == "running")
    return completed, failed, running


def progress_reporter(state_rows, state_lock, stop_event, total_jobs, interval=60):
    """Print a short progress summary to the terminal every interval seconds.

    The summary is written directly to /dev/tty so it appears in the terminal
    even when stdout/stderr are redirected to a log file. If /dev/tty is not
    available, it falls back to stderr.
    """
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

    Only rows with status='completed' are included in the metric averages.
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

    # Convert numeric fields for completed rows.
    for r in completed:
        for k in ("psnr", "ssim", "mse", "time", "t"):
            try:
                r[k] = float(r.get(k, ""))
            except (ValueError, TypeError):
                r[k] = None

    # Group by (method, config_id)
    groups = {}
    for r in completed:
        key = (r["method"], r["config_id"])
        groups.setdefault(key, []).append(r)

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def _std(vals):
        vals = [v for v in vals if v is not None]
        return statistics.stdev(vals) if len(vals) > 1 else 0.0

    stats = []
    for (method, config_id), grp in sorted(groups.items()):
        n = len(grp)
        psnr_vals = [r["psnr"] for r in grp]
        ssim_vals = [r["ssim"] for r in grp]
        mse_vals = [r["mse"] for r in grp]
        t_vals = [r["t"] for r in grp]
        time_vals = [r["time"] for r in grp]
        stats.append({
            "method": method,
            "config_id": config_id,
            "n": n,
            "psnr_mean": _mean(psnr_vals),
            "psnr_std": _std(psnr_vals),
            "ssim_mean": _mean(ssim_vals),
            "ssim_std": _std(ssim_vals),
            "mse_mean": _mean(mse_vals),
            "mse_std": _std(mse_vals),
            "t_mean": _mean(t_vals),
            "t_std": _std(t_vals),
            "time_mean": _mean(time_vals),
            "time_std": _std(time_vals),
        })

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

    # Top 3 per method by PSNR.
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
        # List swept parameters and their values from SWEEP_GRIDS
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
        # Max/min across all completed cells for this method
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

    lines.append("All configurations (averages)")
    lines.append("-" * 80)
    lines.append(f"{'method':<12} {'config_id':<20} {'n':>5} {'psnr':>8} {'ssim':>8} {'mse':>8} {'t(s)':>8} {'time(s)':>8}")
    for s in sorted(stats, key=lambda x: (x["method"], x["config_id"])):
        lines.append(
            f"{s['method']:<12} {s['config_id']:<20} {s['n']:>5} "
            f"{s['psnr_mean']:>8.2f} {s['ssim_mean']:>8.4f} {s['mse_mean']:>8.4f} "
            f"{s['t_mean']:>8.2f} {s['time_mean']:>8.2f}"
        )
    lines.append("")

    text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nReport written to: {output_path}")


def evaluate_job(target, method, config_id):
    """Compute PSNR/SSIM/MSE and imaging time for a finished sweep cell.

    Mirrors the logic in src/compute_metrics.py but adapted to the per-config
    output layout used by the sweep driver.
    """
    if TEST_MODE:
        gt_path = data_root() / target / f"{target}.fits"
        cell_dir = exp_root() / target / f"{target}_{method}" / f"{target}_{method}_{config_id}"
    elif os.path.exists(str(BASE_DIR / "data" / "simulated" / target)):
        gt_path = BASE_DIR / "data" / "simulated" / target / f"{target}.fits"
        cell_dir = BASE_DIR / "experiments" / "simulated" / target / f"{target}_{method}" / f"{target}_{method}_{config_id}"
    elif os.path.exists(str(BASE_DIR / "data" / "archival" / target)):
        gt_path = BASE_DIR / "data" / "archival" / target / f"{target}.fits"
        cell_dir = BASE_DIR / "experiments" / "archival" / target / f"{target}_{method}" / f"{target}_{method}_{config_id}"
    else:
        return {"psnr": "", "ssim": "", "mse": "", "imaging_time": ""}
    recon_path = cell_dir / f"{target}_{method}_{config_id}.fits"
    # CASA and WSClean write logs named after the config_id.  DeepInverse's
    # wrapper writes <target>_deepinverse.log inside the cell directory.
    # R2D2's wrapper writes <target>_r2d2.log; the sweep renames it to the
    # config_id log, but early completed cells only have the wrapper log.
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

    # Runtime extraction mirrors compute_metrics.extract_runtime but the sweep
    # uses the method tag "deepinv" instead of "deepinverse".
    runtime_method = "deepinverse" if method == "deepinv" else method
    t = extract_runtime(str(log_path), runtime_method)
    if np.isnan(t):
        metrics["t"] = ""
    else:
        metrics["t"] = f"{t:.2f}"

    return metrics


def backfill_completed_metrics(state_rows):
    """Compute metrics for completed rows that don't have them yet.

    The wall-clock 'time' column is left untouched: it is measured by the
    runner (subprocess elapsed time) and must not be overwritten by the
    imaging time 't' extracted from the method log.
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

    gt_path = target_dir(BASE_DIR, target) / f"{target}.fits"
    gt = fits.getdata(gt_path).squeeze().astype(np.float64)
    total = gt.sum()

    init_csv(index_csv, ["target", "gt_total_flux"])
    with open(index_csv, "a", newline="") as f:
        csv.writer(f).writerow([target, f"{total:.4f}"])
    return total


# -----------------------------------------------------------------------------
# Preparation step (once per target)
# -----------------------------------------------------------------------------
def prepare_target(target):
    """Ensure GT index entry and DeepInverse weights exist for a target.

    Weights are stored directly inside the target's .mat file in data/simulated/
    (or data/archival/). The old experiments/<target>/<target>_weights/ directory,
    if present, is removed because it only contained temporary WSClean outputs.
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

    # Remove obsolete temporary weights directory from the old layout.
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



# -----------------------------------------------------------------------------
# Formatting helpers for config IDs
# -----------------------------------------------------------------------------
def _fmt(v):
    """Format a value for use in file/folder names. Replaces '.' with 'p'.

    Auto thresholds like 'auto:3' or 'auto:1.5' are shortened to the numeric
    part so that config_ids remain filesystem-safe (no colons) and short.
    """
    s = f"{v}"
    if isinstance(v, str) and s.startswith("auto:"):
        s = s.split(":", 1)[1]
    return s.replace(".", "p")


# -----------------------------------------------------------------------------
# Individual job runners
# -----------------------------------------------------------------------------
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

    # Pass the imagename override so the wrapper writes all files inside the cell dir.
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
            gt_path = _find_gt_path(BASE_DIR, target)
            dirty_prefix = out / f"{target}_wsclean_dirty_pass"
            cache_path = out / ".wsclean_dirty_noise.json"

    cmd = build_wsclean_args(target, pixel_size, str(prefix), params)

    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return 0.0, True

    t0 = time.perf_counter()
    with open(log_path, "w") as logf:
        # Helper logs and manual noise details all go through this function.
        def _log(msg, *args):
            logf.write((msg % args) + "\n")
            logf.flush()

        if _is_auto_threshold(params.get("threshold")) or (
            "threshold" in params and str(params["threshold"]).startswith("AUTO")
        ):
            # Real run: the placeholder was replaced above only in dry-run mode,
            # so this block executes only for actual auto thresholds.
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

            # Rebuild command with the computed threshold.
            cmd = build_wsclean_args(target, pixel_size, str(prefix), params)

        result = subprocess.run(cmd, cwd=out, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0 or not source_path.exists():
        print(f"[error] WSClean {target} {config_id} failed (exit {result.returncode})")
        return elapsed, False

    # Convert Jy/beam -> Jy/pixel in place, then copy to final name without the -image suffix.
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
    """Run the R2D2 imager via scripts/run_r2d2.sh with a fixed configuration.

    The wrapper writes its outputs to the requested cell directory; we then
    rename the generic <target>_r2d2.* files to the sweep's <target>_r2d2_<config_id>.*
    naming convention so that evaluate_job can locate them.
    """
    out = exp_target_dir(BASE_DIR, target)
    cell_dir = out / f"{target}_r2d2" / f"{target}_r2d2_{config_id}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    # Generic names produced by the wrapper inside the cell directory.
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

    # Persist the captured log with the sweep naming convention.
    with open(final_log, "w") as logf:
        logf.write(result.stdout)

    if result.returncode != 0 or not source_fits.exists():
        print(f"[error] R2D2 {target} {config_id} failed (exit {result.returncode})")
        return elapsed, False

    shutil.copy2(source_fits, final_fits)
    return elapsed, True


def run_deepinv_warmup(target, state_csv, state_lock, state_rows):
    """Run a single DeepInverse warm-up job to compute and cache the operator norm.

    This job also serves as the global CUDA/kernel warm-up for the target. Its
    elapsed wall-clock time is recorded in sweep_state.csv with config_id='warmup'
    and empty metric columns.
    """
    out = exp_target_dir(BASE_DIR, target)
    cell_dir = out / f"{target}_deepinv" / f"{target}_deepinv_warmup"
    cell_dir.mkdir(parents=True, exist_ok=True)

    source_path = cell_dir / f"{target}_deepinverse.fits"
    final_path = cell_dir / f"{target}_deepinv_warmup.fits"

    # Base config without opnorm and without skipping warm-up.
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


# -----------------------------------------------------------------------------
# Target-level work dispatchers
# -----------------------------------------------------------------------------
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

    The operator norm for DeepInverse is computed via a warm-up run immediately
    before the first DeepInverse config of the target if it is not already cached.
    The cache file is kept on disk so a resumed sweep can reuse it.
    """
    # R2D2: single fixed configuration dispatched to the GPU worker.
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

    # Warm-up / opnorm cache is computed here, right before the target's configs.
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



# -----------------------------------------------------------------------------
# Workers
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
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

    global TEST_MODE
    if args.test:
        TEST_MODE = True
        os.environ["RI_BENCH_TEST"] = "1"
        if args.cpu_workers != 1 or args.gpu_workers != 1:
            print("[test] forcing --cpu-workers 1 and --gpu-workers 1")
        args.cpu_workers = 1
        args.gpu_workers = 1

    if args.targets is None:
        args.targets = str(BASE_DIR / "scripts" / ("test_targets.txt" if TEST_MODE else "sweep_targets.txt"))
    if args.sweep_config is None:
        args.sweep_config = str(BASE_DIR / "scripts" / ("test_config.txt" if TEST_MODE else "sweep_config.txt"))

    sweep_config_file = Path(args.sweep_config)
    if not sweep_config_file.exists():
        print(f"Sweep config file not found: {sweep_config_file}")
        sys.exit(1)

    # Load sweep grids into the global read-only structure used by workers and reports.
    global SWEEP_GRIDS
    SWEEP_GRIDS = build_sweep_grids(sweep_config_file)
    print("Loaded sweep methods:", list(SWEEP_GRIDS.keys()))

    if TEST_MODE:
        # Guardrail: the test set is evaluated once with a single winning config
        # per method, not swept. Abort if any method expands to more than one cell.
        for method, combos in SWEEP_GRIDS.items():
            if len(combos) != 1:
                print(f"[test] ERROR: method '{method}' has {len(combos)} configs in test mode; exactly 1 is required.")
                sys.exit(1)

    if args.evaluate:
        state_csv = sweep_dir(BASE_DIR) / ("test_state.csv" if TEST_MODE else "sweep_state.csv")
        output_path = sweep_dir(BASE_DIR) / ("test_results.txt" if TEST_MODE else "sweep_results.txt")
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
    state_csv = sweep_root / ("test_state.csv" if TEST_MODE else "sweep_state.csv")
    state_lock = threading.Lock()
    state_rows = load_state(state_csv)
    if not args.dry_run:
        backfill_completed_metrics(state_rows)
        save_state(state_csv, state_lock, state_rows)  # persist cleaning of stale 'running' rows + metrics backfill

    # Per-target locks prevent two workers from colliding on the same target.
    # With the two-phase scheduler CPU jobs and GPU jobs never overlap, but
    # multiple CPU (or GPU) workers may still pick the same target, so we keep
    # the locks for safety.
    cpu_target_locks = {t: threading.Lock() for t in targets}
    gpu_target_locks = {t: threading.Lock() for t in targets}

    # Preparation step: GT flux index and DeepInverse weights.
    # Skip during dry-run to avoid generating any data.
    if not args.dry_run and not args.no_prepare:
        for target in targets:
            prepare_target(target)

    # Count the actual number of jobs that would run (accounting for resumption).
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

    # Queues: one for CPU targets, one for GPU targets.
    from queue import Queue
    cpu_queue = Queue()
    gpu_queue = Queue()
    for target in targets:
        cpu_queue.put(target)
        gpu_queue.put(target)

    # Sentinel values to stop workers.
    for _ in range(args.cpu_workers):
        cpu_queue.put(None)
    for _ in range(args.gpu_workers):
        gpu_queue.put(None)

    t_start = time.perf_counter()

    # Progress reporter: prints to terminal every 60s without polluting stdout log.
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

    # Phase 1: CPU-bound jobs (CASA + WSClean). Run these first so that GPU-
    # bound DeepInverse jobs do not contend for CPU/memory/NUFFT resources.
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

    # Phase 2: GPU-bound jobs (DeepInverse). Start only after CPU phase is done.
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
