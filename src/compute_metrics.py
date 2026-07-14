import os
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def getpaths(target, method):
    # Checks if the target is simulated or archival
    if os.path.exists(os.path.join("data", "simulated", target)):
        ground_truth_path = os.path.join("data", "simulated", target, f"{target}.fits")
        reconstruction_path = os.path.join("experiments", "simulated", target, f"{target}_{method}", f"{target}_{method}.fits")
        log_path = os.path.join("experiments", "simulated", target, f"{target}_{method}", f"{target}_{method}.log")
        simulated = True
        base_exp_dir = os.path.join("experiments", "simulated", target)
    else:
        ground_truth_path = os.path.join("data", "archival", target, f"{target}.fits")
        reconstruction_path = os.path.join("experiments", "archival", target, f"{target}_{method}", f"{target}_{method}.fits")
        log_path = os.path.join("experiments", "archival", target, f"{target}_{method}", f"{target}_{method}.log")
        simulated = False
        base_exp_dir = os.path.join("experiments", "archival", target)

    csv_path = os.path.join("results", "tables", f"{target}.csv")

    return ground_truth_path, reconstruction_path, log_path, simulated, csv_path, base_exp_dir

# def normalize_to_01(data):
#     """Normalize data to the range [0, 1] using min-max normalization."""
#     dmin = data.min()
#     dmax = data.max()
#     if dmax - dmin == 0:
#         return np.zeros_like(data)
#     return (data - dmin) / (dmax - dmin)

def normalize_to_01(data):
    """Normalize data to [0, 1] via standard min-max normalization after clipping negatives."""
    data = np.clip(data, 0, None)
    dmin = data.min()
    dmax = data.max()
    if dmax - dmin == 0:
        return np.zeros_like(data)
    return (data - dmin) / (dmax - dmin)

def compute_ground_truth_metrics(reconstruction, ground_truth):
    """Calculate core image reconstruction metrics."""
    data_range = ground_truth.max() - ground_truth.min()
    return {
        "psnr": round(psnr(reconstruction, ground_truth, data_range=data_range), 2),
        "ssim": round(ssim(reconstruction, ground_truth, data_range=data_range), 4),
        "mse": round(np.mean((reconstruction - ground_truth)**2), 4)
    }

def compute_blind_metrics(reconstruction):
    """Calculate blind image reconstruction metrics."""
    return {}

def extract_runtime(log_path, method):
    if not os.path.exists(log_path):
        return np.nan
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if method == 'wsclean' and "deconvolution:" in line:
                    # Line looks like: Inversion: 00:00:00.374290, prediction: 00:00:00.226456, deconvolution: 00:00:00.050137
                    total_ws_time = 0.0
                    parts = line.split(',')
                    for p in parts:
                        time_str = p.split(':')[1:]  # e.g., [" 00", "00", "00.374290"]
                        time_str = ":".join(time_str).strip()
                        h, m, s = time_str.split(':')
                        total_ws_time += float(h)*3600 + float(m)*60 + float(s)
                    return total_ws_time
                elif method == 'r2d2' and "Imaging finished in" in line:
                    time_str = line.split("iterations in ")[1].split(" sec")[0].strip()
                    return float(time_str)
                elif method == 'deepinverse' and "Imaging time:" in line:
                    time_str = line.split("Imaging time:")[1].split("seconds")[0].strip()
                    return float(time_str)
                elif method == 'casa' and "Imaging time:" in line:
                    time_str = line.split("Imaging time:")[1].split("seconds")[0].strip()
                    return float(time_str)
    except Exception as e:
        pass
    return np.nan

def calculate_errormap(recon_path, gt_data, recon_data, target, method):
    """Calculate residual image."""
    errormap = recon_data - gt_data
    recon_dir = os.path.dirname(recon_path)
    errormap_path = os.path.join(recon_dir, f"{target}_{method}_errormap.fits")
    hdr = fits.getheader(recon_path)
    fits.writeto(errormap_path, errormap, hdr, overwrite=True)

def evaluate_target_all_methods(target):
    """Process a target by comparing reconstructions directly against the original ground truth."""
    methods = ['casa', 'wsclean', 'deepinverse', 'r2d2']
    available_methods = []

    # Check if target exists
    if not (os.path.exists(os.path.join("data", "simulated", target)) or 
            os.path.exists(os.path.join("data", "archival", target))):
        print(f"Target {target} not found in data directories.")
        return

    # Grab basic paths
    ground_truth_path, _, _, simulated, csv_path, base_exp_dir = getpaths(target, "dummy")

    for method in methods:
        _, recon_path, _, _, _, _ = getpaths(target, method)
        if os.path.exists(recon_path):
            available_methods.append(method)

    if not available_methods:
        print(f"No valid reconstructions found for target {target}.")
        return

    print(f"\n--- Processing target: {target} ---")
    print(f"Found methods: {available_methods}")

    # Load ground truth (already in [0,1])
    has_gt = os.path.exists(ground_truth_path)
    if has_gt:
        gt_data = fits.getdata(ground_truth_path).squeeze().astype(np.float64)
        print(f"Ground truth range: [{gt_data.min():.4f}, {gt_data.max():.4f}]")

    # Prepare DataFrame collection for CSV
    rows = []

    for method in available_methods:
        print(f"Evaluating {method}...")
        _, recon_path, log_path, _, _, _ = getpaths(target, method)

        recon_data = fits.getdata(recon_path).squeeze().astype(np.float64)
        print(f"  Reconstruction range: [{recon_data.min():.4e}, {recon_data.max():.4e}]")

        # Normalization for the metrics
        # recon_data = normalize_to_01(recon_data)

        m = {}

        if has_gt:
            # Generate error map
            calculate_errormap(recon_path, gt_data, recon_data, target, method)
            if simulated:
                m = compute_ground_truth_metrics(recon_data, gt_data)

        if not simulated or not has_gt:
            m = compute_blind_metrics(recon_data)

        m["t"] = extract_runtime(log_path, method)

        # Create row
        row = {
            "target": target,
            "simulated": simulated,
            "method": method,
            **m
        }
        rows.append(row)
        print(f"  PSNR: {m.get('psnr')} | SSIM: {m.get('ssim')} | MSE: {m.get('mse')} | t: {m.get('t')}")

    # Save results to CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_new = pd.DataFrame(rows)

    with open(csv_path, 'w') as f:
        df_new.to_csv(f, index=False)

def discover_targets():
    """Scan data directories to find all available targets."""
    targets = []
    for category in ['simulated', 'archival']:
        path = os.path.join('data', category)
        if os.path.exists(path):
            for entry in os.listdir(path):
                if os.path.isdir(os.path.join(path, entry)):
                    targets.append(entry)
    return sorted(list(set(targets)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RI Benchmark Evaluation")
    parser.add_argument("--target", type=str, help="Target name (e.g., disk_0000). If omitted, runs all available targets.")
    args = parser.parse_args()

    if args.target:
        evaluate_target_all_methods(args.target)
    else:
        targets = discover_targets()
        print(f"Found {len(targets)} targets to process: {targets}")
        for t in targets:
            evaluate_target_all_methods(t)