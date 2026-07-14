import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.transforms import calculate_pixel_size
import deepinv as dinv
import torch
import os
import argparse
import scipy.io
import numpy as np
import time
from tqdm import tqdm
from deepinv.physics import RadioInterferometry
from deepinv.utils.plotting import plot, plot_curves, scatter_plot, plot_inset
from deepinv.utils import load_np_url, get_image_url, get_degradation_url
from deepinv.utils.tensorlist import dirac_like


def main(target):  
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')) # Project root
    # Check if target is simulated or archival
    if os.path.exists(os.path.join(base_dir,'data','simulated',target)):
        vis_path = os.path.join(base_dir, 'data','simulated',target, target + '.mat')
        out_dir = os.path.join(base_dir, 'experiments', 'simulated', target, target + '_deepinverse') + '/'
    else:
        vis_path = os.path.join(base_dir, 'data','archival',target, target + '.mat')
        out_dir = os.path.join(base_dir, 'experiments', 'archival', target, target + '_deepinverse') + '/'
    
    def savefits(tensor, name=""):
        from astropy.io import fits
        import scipy.io
        os.makedirs(out_dir, exist_ok=True)
        if isinstance(tensor, (list, tuple)):
            tensor = tensor[0]
        if hasattr(tensor, 'detach'):
            tensor = tensor.detach().cpu().numpy()
        if np.iscomplexobj(tensor):
            tensor = np.abs(tensor)
        tensor = np.squeeze(tensor)  # (H, W)
            
        filename = f"{target}_deepinverse_{name}.fits" if name else f"{target}_deepinverse.fits"
        filepath = os.path.join(out_dir, filename)
        hdu = fits.PrimaryHDU(tensor)
        
        # Add basic WCS and beam headers
        try:
            mat_data = scipy.io.loadmat(vis_path)
            pixel_size = float(mat_data.get('nominal_pixelsize').flatten()[0])
            hdu.header['BMAJ'] = ( 2 * pixel_size) / 3600.0
            hdu.header['BMIN'] = ( 2 * pixel_size) / 3600.0
            hdu.header['BPA'] = 0.0
            hdu.header['CDELT1'] = -pixel_size / 3600.0
            hdu.header['CDELT2'] = pixel_size / 3600.0
            hdu.header['BUNIT'] = 'Jy/pixel'
        except Exception as e:
            print(f"Warning: could not add header keywords: {e}")

        hdu.writeto(filepath, overwrite=True)
        
        # Normalize the main reconstruction to [0, 1] (skip dirty/psf)
        if not name:
            from src.utils.transforms import normalize_fits_peak
            normalize_fits_peak(filepath)
        
        print(f"Saved: {filepath}")
        

    # Clear previous results
    import glob
    for f in glob.glob(os.path.join(out_dir, '*')):
        os.remove(f)

    # Load device
    device = dinv.utils.get_device()

    # Convert mat to npy
    mattonpy(vis_path)
    
    # Load npy. Important to specify dtype, otherwise it loads as float32
    data = dinv.utils.load_np(vis_path.replace('.mat', '.npy'), dtype=np.complex64, device=device)
    uv = dinv.utils.load_np(vis_path.replace('.mat', '_uv.npy'), dtype=np.float32, device=device)
    briggs = dinv.utils.load_np(vis_path.replace('.mat', '_briggs.npy'), dtype=np.float32, device=device)
    nW = dinv.utils.load_np(vis_path.replace('.mat', '_nW.npy'), dtype=np.float32, device=device)
    
    print(f"Data: {data.dtype} {data.shape}")
    print(f"UV: {uv.dtype} {uv.shape}")
    print(f"Briggs weights: {briggs.dtype} {briggs.shape}")
    print(f"nW (1/tau): {nW.dtype} {nW.shape}")

    # Combine Briggs weighting and natural weighting (nWimag / tau in the tutorial)
    briggs = briggs.view(1, 1, -1).to(device)
    nW = nW.view(1, 1, -1).to(device)
    nWimag = briggs * nW
    
    # Apply natural + Briggs weighting to measurements
    y_raw = data.view(1, 1, -1).to(device)
    y = y_raw * nWimag
    
    # Setup the RadioInterferometry forward operator
    physics = RadioInterferometry(
        img_size=(64, 64),
        samples_loc=uv.permute((1, 0)),
        real_projection=True,
        device=device,
    )
    # Add image weighting to the sensing operator
    physics.setWeight(nWimag)
    
    # Central Dirac pulse for PSF
    mock = torch.randint(5, (1, 1, 64, 64))
    dirac = dirac_like(mock).to(device)
    
    # Compute PSF
    PSF = physics.A_adjoint(physics.A(dirac))
    savefits(PSF, 'psf')
   
    # Compute dirty image (single W via operator adjoint: Φᴴ · W · y_raw)
    x_hat = physics.A_adjoint(y_raw.view(1, 1, -1).to(device))
    savefits(x_hat, 'dirty')
        
    # Initialization for FISTA: use double-weighted Aᴴ·y_w = Φᴴ·W²·y_raw
    # This matches the gradient of ||Ax - y_w||² at x=0
    x_init_fista = physics.A_adjoint(y).to(device)
        
    # Compute operator norm exactly as in DeepInv tutorial
    opnorm = physics.compute_sqnorm(
        torch.randn((1, 1, 64, 64), device=device),
        max_iter=100,
        tol=1e-6,
        verbose=False,
    ).item()
    print(f"Operator norm (raw): {opnorm}")
    
    # # Normalize weights and data so operator norm ≈ 1.0 across all datasets
    # # This preserves the *relative* Briggs weighting but standardizes the scale
    # scale = 1.0 / (opnorm ** 0.5)
    # nWimag_norm = nWimag * scale
    # y = y * scale
    # physics.setWeight(nWimag_norm)
    
    # # Recompute dirty image with normalized weights
    # x_hat = physics.A_adjoint(y).to(device)

    from deepinv.optim.data_fidelity import L2
    from deepinv.optim.prior import WaveletPrior
    
    # Select the data fidelity term
    data_fidelity = L2()
    # Specify the prior (we redefine it with a smaller number of iteration for faster computation)
    wv_list = ["db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8"]
    prior = WaveletPrior(level=3, wv=wv_list, p=1, device=device, clamp_min=0)
   
    from deepinv.optim.optimizers import optim_builder
    # Logging parameters
    verbose = True
    plot_convergence_metrics = False # Log performance metrics

    # Algo parameters
    stepsize = 1.0 / (1.5 * opnorm)
    lamb = 2e-3 * opnorm  # Regularization parameter
   
    # # Algo parameters (now consistent across all datasets since opnorm ≈ 1)
    # stepsize = 1.0 / 1.5  # 1 / (1.5 * 1.0)
    # lamb = 1e-5  # Fixed regularization parameter
    
    params_algo = {"stepsize": stepsize, "lambda": lamb, "a": 3}
    # Increase max_iter to allow FISTA to fully converge on the sharp edges
    max_iter = 50
    early_stop = True
   
    def custom_init(y, physics):
        x_init = torch.clamp(physics.A_adjoint(y), 0)
        return {"est": (x_init, x_init)}

    model = optim_builder(
        iteration="FISTA",
        prior=prior,
        data_fidelity=data_fidelity,
        early_stop=early_stop,
        max_iter=max_iter,
        verbose=verbose,
        params_algo=params_algo,
        custom_init=custom_init,
    )

    init = torch.clamp(x_init_fista, 0), torch.clamp(x_init_fista, 0) # FISTA initialization (reuse dirty image)
    
    # Warm-up run
    _ = model(y, physics, init=init)
    
    # Runtime measurement
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    # FISTA reconstruction (timed)
    x_model = model(y, physics, init=init)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    # runtime = end_time - start_time
    # Read weight computation time from the weights log if it exists
    weight_time = 0.0
    if os.path.exists(os.path.join(base_dir, 'data', 'simulated', target)):
        weights_log = os.path.join(base_dir, 'experiments', 'simulated', target, f"{target}_weights", f"{target}_wsclean.log")
    else:
        weights_log = os.path.join(base_dir, 'experiments', 'archival', target, f"{target}_weights", f"{target}_wsclean.log")
    
    if os.path.exists(weights_log):
        try:
            with open(weights_log, 'r', encoding='utf-8', errors='ignore') as wf:
                for line in wf:
                    if "Execution time:" in line:
                        weight_time = float(line.split("Execution time:")[1].split("seconds")[0].strip())
        except Exception as e:
            print(f"Warning: could not read weight computation time: {e}")
            
    runtime = (end_time - start_time) + weight_time
    imaging_time = end_time - start_time
    
    savefits(x_model, '')

    # Log configuration and execution time
    log_filepath = os.path.join(out_dir, f"{target}_deepinverse.log")
    with open(log_filepath, "w") as f:
        f.write("DeepInv Configuration:\n")
        f.write("----------------------\n")
        f.write(f"Operator: RadioInterferometry\n")
        f.write(f"Image size: (64, 64)\n")
        f.write(f"Real projection: True\n")
        f.write(f"Device: {device}\n")
        f.write(f"Operator norm (raw): {opnorm}\n")
        # f.write(f"Normalization scale: {scale} (1/sqrt(opnorm))\n\n")
        
        f.write("Optimization / Prior:\n")
        f.write("---------------------\n")
        f.write(f"Iteration: FISTA\n")
        f.write(f"Data Fidelity: L2\n")
        f.write(f"Prior: WaveletPrior (level=3, wv={wv_list}, p=1, clamp_min=0)\n")
        f.write(f"Max iterations: {max_iter}\n")
        f.write(f"Early stop: {early_stop}\n")
        f.write(f"Stepsize: {stepsize} (1.0 / 1.5, normalized opnorm ~= 1)\n")
        f.write(f"Lambda: {lamb} (fixed, dataset-independent)\n")
        f.write(f"Params algo: {params_algo}\n\n")
        
        f.write(f"Imaging time: {imaging_time:.2f} seconds\n")
        f.write(f"Execution time: {runtime:.2f} seconds\n")
        
    print(f"Imaging time: {imaging_time:.2f} seconds")
    print(f"Execution time: {runtime:.2f} seconds")

    # Cleanup
    npy_file = vis_path.replace('.mat', '.npy')
    uv_npy_file = vis_path.replace('.mat', '_uv.npy')
    briggs_npy_file = vis_path.replace('.mat', '_briggs.npy')
    nW_npy_file = vis_path.replace('.mat', '_nW.npy')
    if os.path.exists(npy_file): os.remove(npy_file)
    if os.path.exists(uv_npy_file): os.remove(uv_npy_file)
    if os.path.exists(briggs_npy_file): os.remove(briggs_npy_file)
    if os.path.exists(nW_npy_file): os.remove(nW_npy_file)


def normalize_uv(uv):
    """Normalize UV coordinates to [-pi, pi]."""
    max_val = np.abs(uv).max()
    if max_val > 0:
        uv = (uv / max_val) * np.pi
    return uv


def mattonpy(mat):
    mat_data = scipy.io.loadmat(mat)
    vis_path = mat.replace('.mat', '.MS')
    y = mat_data.get('DATA', mat_data.get('y'))
    u = mat_data.get('u')
    v = mat_data.get('v')
    
    pixel_size = calculate_pixel_size(vis_path,n=3)  # arcsec
    cell_size_rad = pixel_size * np.pi / (180.0 * 3600.0)
    u = u * (2.0 * np.pi * cell_size_rad)
    v = v * (2.0 * np.pi * cell_size_rad)
    uv = np.concatenate((-v, u), axis=1)
    # uv = normalize_uv(uv)
    
    # Save arrays
    np.save(mat.replace('.mat', '.npy'), y)
    np.save(mat.replace('.mat', '_uv.npy'), uv)
    
    # Save Briggs weights (from wsclean imaging weights)
    briggs = mat_data.get('weight')
    if briggs is None:
        briggs = np.ones((y.shape[0], 1), dtype=np.float32)
    np.save(mat.replace('.mat', '_briggs.npy'), briggs)
    
    # Save natural weights nW = 1/tau (inverse noise std from simulator)
    nW = mat_data.get('nW')
    if nW is None:
        nW = np.ones((y.shape[0], 1), dtype=np.float32)
    np.save(mat.replace('.mat', '_nW.npy'), nW)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MAT to NPY.')
    parser.add_argument('--target', type=str, required=True, help='Target target name')
    args = parser.parse_args()
    
    main(args.target)