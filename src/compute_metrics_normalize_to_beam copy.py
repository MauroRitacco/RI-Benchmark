import os
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
from utils.transforms import smooth_to_beam, convert_jybeam_to_jypixel, convert_jypixel_to_jybeam, formatfits
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

def extract_runtime(log_path):
    if not os.path.exists(log_path):
        return np.nan
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if "Execution time:" in line:
                    time_str = line.split("Execution time:")[1].split("seconds")[0].strip()
                    return float(time_str)
    except Exception as e:
        pass
    return np.nan

def calculate_errormap(smoothed_recon_path, smoothed_gt_path, target, method):
    """Calculate residual image from smoothed maps."""
    reconstruction = fits.getdata(smoothed_recon_path).squeeze()
    ground_truth = fits.getdata(smoothed_gt_path).squeeze()
    errormap = reconstruction - ground_truth
    
    recon_dir = os.path.dirname(smoothed_recon_path)
    errormap_path = os.path.join(recon_dir, f"{target}_{method}_errormap.fits")
    hdr = fits.getheader(smoothed_recon_path)
    fits.writeto(errormap_path, errormap, hdr, overwrite=True)

def evaluate_target_all_methods(target):
    """Process a target by smoothing all available methods to the biggest beam."""
    methods = ['casa', 'wsclean', 'deepinverse', 'r2d2']
    available_methods = []
    
    # 1. Discover available methods and gather their beam sizes
    max_area = -1.0
    biggest_beam = None
    
    # Check if target exists
    if not (os.path.exists(os.path.join("data", "simulated", target)) or 
            os.path.exists(os.path.join("data", "archival", target))):
        print(f"Target {target} not found in data directories.")
        return

    # Grab basic paths to figure out where to look
    ground_truth_path, _, _, simulated, csv_path, base_exp_dir = getpaths(target, "dummy")
    
    max_bmaj = 0.0
    
    for method in methods:
        _, recon_path, _, _, _, _ = getpaths(target, method)
        if os.path.exists(recon_path):
            try:
                hdr = fits.getheader(recon_path)
                bmaj = hdr.get('BMAJ', 0)
                bmin = hdr.get('BMIN', 0)
                bpa = hdr.get('BPA', 0)
                
                # We only consider methods that have valid beam information
                if bmaj > 0 and bmin > 0:
                    available_methods.append(method)
                    if bmaj > max_bmaj:
                        max_bmaj = bmaj
            except Exception as e:
                print(f"Error reading header of {recon_path}: {e}")

    if not available_methods:
        print(f"No valid reconstructions with beam parameters found for target {target}.")
        return

    # Construct the common circular bounding beam
    biggest_beam = {
        'method': 'Common_Circular_Beam',
        'bmaj_deg': max_bmaj,
        'bmin_deg': max_bmaj,
        'bpa_deg': 0.0
    }

    print(f"\n--- Processing target: {target} ---")
    print(f"Found methods: {available_methods}")
    
    print(f"Common Bounding Beam:")
    print(f"  BMAJ: {biggest_beam['bmaj_deg']*3600:.4f} arcsec")
    print(f"  BMIN: {biggest_beam['bmin_deg']*3600:.4f} arcsec")
    print(f"  BPA:  {biggest_beam['bpa_deg']:.2f} deg")

    # 2. Smooth the Ground Truth to the biggest beam if it exists
    has_gt = os.path.exists(ground_truth_path)
    smoothed_gt_path = os.path.join(base_exp_dir, f"{target}_smoothed.fits")
    os.makedirs(base_exp_dir, exist_ok=True)
    
    if has_gt:
        # Make sure Ground Truth has WCS headers
        gt_hdr = fits.getheader(ground_truth_path)
        if 'CTYPE1' not in gt_hdr:
            ms_path = os.path.join(os.path.dirname(ground_truth_path), f"{target}.MS")
            formatfits(ground_truth_path, ms_path)

        print("Smoothing ground truth...")
        smooth_to_beam(ground_truth_path, smoothed_gt_path, 
                       biggest_beam['bmaj_deg'], biggest_beam['bmin_deg'], biggest_beam['bpa_deg'])
    else:
        print("No ground truth found (likely archival data). Skipping GT smoothing.")

    # Prepare DataFrame collection for CSV
    rows = []

    # 3. Process each available method
    for method in available_methods:
        print(f"Evaluating {method}...")
        _, recon_path, log_path, _, _, _ = getpaths(target, method)
        
        # Ensure reconstruction has WCS headers before smoothing
        recon_hdr = fits.getheader(recon_path)
        if 'CTYPE1' not in recon_hdr:
            print(f"Warning: {method} reconstruction missing WCS headers. Formatting...")
            ms_path = os.path.join(os.path.dirname(ground_truth_path), f"{target}.MS") if has_gt else os.path.join("data", "archival", target, f"{target}.MS")
            formatfits(recon_path, ms_path)

        # Convert Jy/beam to Jy/pixel if necessary before smoothing (though usually we smooth first)
        # imsmooth requires standard units. We'll convert after smoothing to match GT units.
        smoothed_recon_path = os.path.join(os.path.dirname(recon_path), f"{target}_{method}_smoothed.fits")
        
        # If the reconstruction was previously converted to Jy/pixel by generation scripts, 
        # restore it to Jy/beam before passing to CASA imsmooth to prevent corrupted scaling.
        hdr = fits.getheader(recon_path)
        if hdr.get('BUNIT') == 'Jy/pixel' and 'BMAJ' in hdr:
            convert_jypixel_to_jybeam(recon_path)

        # Smooth the method reconstruction to the biggest beam
        smooth_to_beam(recon_path, smoothed_recon_path, 
                       biggest_beam['bmaj_deg'], biggest_beam['bmin_deg'], biggest_beam['bpa_deg'])

        recon_data = fits.getdata(smoothed_recon_path).squeeze()
        m = {}

        if has_gt:
            # Generate error map
            calculate_errormap(smoothed_recon_path, smoothed_gt_path, target, method)
            # Load GT data to compute metrics
            gt_data = fits.getdata(smoothed_gt_path).squeeze()
            if simulated:
                m = compute_ground_truth_metrics(recon_data, gt_data)
        
        if not simulated or not has_gt:
            m = compute_blind_metrics(recon_data)
            
        m["t"] = extract_runtime(log_path)
        
        # Create row
        row = {
            "target": target,
            "simulated": simulated,
            "method": method,
            **m
        }
        rows.append(row)
        print(f"  PSNR: {m.get('psnr')} | SSIM: {m.get('ssim')} | MSE: {m.get('mse')} | t: {m.get('t')}")

    # 4. Save results to CSV (Overwrite existing target entries)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_new = pd.DataFrame(rows)
    
    comment_line = f"# Biggest Beam: method={biggest_beam['method']}, BMAJ={biggest_beam['bmaj_deg']*3600:.4f} arcsec, BMIN={biggest_beam['bmin_deg']*3600:.4f} arcsec, BPA={biggest_beam['bpa_deg']:.2f} deg\n"
    
    with open(csv_path, 'w') as f:
        f.write(comment_line)
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
    parser.add_argument("--method", type=str, help="Legacy argument: Method name (casa, r2d2, deepinv, wsclean). This is now ignored, as the script evaluates all available methods for the target.")
    args = parser.parse_args()

    if args.target:
        evaluate_target_all_methods(args.target)
    else:
        targets = discover_targets()
        print(f"Found {len(targets)} targets to process: {targets}")
        for t in targets:
            evaluate_target_all_methods(t)