import os
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
from utils.transforms import smoothimage, convert_jybeam_to_jypixel
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def getpaths(target,method):
        # Checks if the target is simulated or archival
    if os.path.exists(os.path.join("data","simulated",target)):
        ground_truth_path = os.path.join("data","simulated", target, f"{target}.fits")
        reconstruction_path = os.path.join("experiments", 'simulated', target, f"{target}_{method}", f"{target}_{method}.fits")
        log_path = os.path.join("experiments", 'simulated', target, f"{target}_{method}", f"{target}_{method}.log")
        simulated = True
    else:
        ground_truth_path = os.path.join("data","archival", target, f"{target}.fits")
        reconstruction_path = os.path.join("experiments", 'archival', target, f"{target}_{method}", f"{target}_{method}.fits")
        log_path = os.path.join("experiments", 'archival', target, f"{target}_{method}", f"{target}_{method}.log")
        simulated = False

    csv_path = os.path.join("results", "tables", f"{target}.csv")

    return ground_truth_path, reconstruction_path, log_path, simulated, csv_path

def compute_ground_truth_metrics(reconstruction, ground_truth):
    """Calculate core image reconstruction metrics."""
    # Define data range based on ground truth for proper scaling
    data_range = ground_truth.max() - ground_truth.min()
    
    return {
        "psnr": round(psnr(reconstruction, ground_truth, data_range=data_range), 2),
        "ssim": round(ssim(reconstruction, ground_truth, data_range=data_range), 4),
        "mse": round(np.mean((reconstruction - ground_truth)**2), 4)
    }

def compute_blind_metrics(reconstruction):
    """Calculate blind image reconstruction metrics."""
    pass

def evaluate_experiment(target, method):
    """Load images, compute metrics, and log results to CSV."""
    ground_truth_path, reconstruction_path, log_path, simulated, csv_path = getpaths(target, method)
    

    if fits.getheader(reconstruction_path).get('BUNIT') == 'Jy/beam':
        convert_jybeam_to_jypixel(reconstruction_path)

    smoothed_gt_path = smoothimage(reconstruction_path, ground_truth_path)


    # Compute errormap
    calculate_errormap(reconstruction_path, smoothed_gt_path, target, method)

    # Load image data as numpy arrays
    reconstruction = fits.getdata(reconstruction_path).squeeze()
    smoothed_gt = fits.getdata(smoothed_gt_path).squeeze()



    # Run mathematical computation
    if simulated:
        m = compute_ground_truth_metrics(reconstruction, smoothed_gt)
    else:
        m = compute_blind_metrics(reconstruction_path, ) #TODO: Implement blind metrics
    
    # Extract runtime
    m["t"] = extract_runtime(log_path)

    # Log results
    row = {"target": target,"simulated": simulated, "method": method, **m}
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_new = pd.DataFrame([row])
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df[~((df['target'] == target) & (df['method'] == method))]
        df = pd.concat([df, df_new], ignore_index=True)
        df.to_csv(csv_path, index=False)
    else:
        df_new.to_csv(csv_path, index=False)
    
    return row
    
def extract_runtime(log_path):
    """Extract runtime from the specific format: 'Execution time: float seconds'"""
    if not os.path.exists(log_path):
        print(f"Log not found: {log_path}")
        return np.nan
        
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if "Execution time:" in line:
                    time_str = line.split("Execution time:")[1].split("seconds")[0].strip()
                    return float(time_str)
    except Exception as e:
        print(f"Error reading log {log_path}: {e}")
        
    return np.nan

def calculate_errormap(reconstruction_path, ground_truth_path, target, method):
    """Calculate residual image."""
    reconstruction = fits.getdata(reconstruction_path).squeeze()
    ground_truth = fits.getdata(ground_truth_path).squeeze()
    errormap = reconstruction - ground_truth
    errormap_path = os.path.join("experiments", "simulated", target, f"{target}_{method}", f"{target}_{method}_errormap.fits")
    hdr=fits.getheader(reconstruction_path)
    fits.writeto(errormap_path, errormap, hdr, overwrite=True)


if __name__ == "__main__":
    # 1. Setup command line arguments
    parser = argparse.ArgumentParser(description="Run RI Benchmark Evaluation")
    parser.add_argument("--target", type=str, help="Target name (e.g., disk_0000)")
    parser.add_argument("--method", type=str, help="Method name (casa, r2d2, deepinv, wsclean)")
    
    args = parser.parse_args()

    # 2. Run evaluation
    if args.target and args.method:
        print(f"Evaluating {args.method} on {args.target}...")
        result = evaluate_experiment(args.target, args.method)
        if result:
            print(f"Done! PSNR: {result['psnr']:.2f} | SSIM: {result['ssim']:.4f} | MSE: {result['mse']:.4f} | t: {result['t']:.2f}")
    else:
        print("Please provide --target and --method")