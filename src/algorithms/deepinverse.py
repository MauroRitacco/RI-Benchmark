import sys
import os
import json
import argparse
import glob
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import torch
import scipy.io
import deepinv as dinv
from tqdm import tqdm
from astropy.io import fits
from deepinv.physics import RadioInterferometry
from deepinv.utils.plotting import plot, plot_curves, scatter_plot, plot_inset
from deepinv.utils import load_np_url, get_image_url, get_degradation_url
from deepinv.utils.tensorlist import dirac_like
from src.utils.transforms import calculate_pixel_size


def main(target, config_json=None, n=1):  
    # Default configuration
    cfg = {
        "img_size": [64, 64],
        "real_projection": True,
        "prior": "wavelet",
        "prior_level": 3,
        "prior_wavelets": ["db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8"],
        "prior_p": 1,
        "prior_clamp_min": 0,
        "stepsize_factor": 1.5,
        "lambda_factor": 1e-3,
        "params_algo_a": 3,
        "max_iter": 50,
        "early_stop": True,
        "thres_conv": 1e-5,
        # Optional cached operator norm (avoids recomputing it for every config).
        "opnorm": None,
        # Skip the FISTA warm-up run. Useful when a global warm-up has already been done.
        "skip_warmup": False,
    }
    if config_json:
        user_cfg = json.loads(config_json)
        cfg.update(user_cfg)

    # Optional per-cell output directory override (used by the sweep driver).
    # Keep default behaviour when it is not supplied.
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')) # Project root
    if os.environ.get("RI_BENCH_TEST") == "1":
        sim_dir = os.path.join(base_dir, 'data', 'simulated', 'test')
        exp_dir = os.path.join(base_dir, 'experiments', 'test')
    else:
        sim_dir = os.path.join(base_dir, 'data', 'simulated')
        exp_dir = os.path.join(base_dir, 'experiments', 'simulated')
    if os.path.exists(os.path.join(sim_dir, target)):
        vis_path = os.path.join(sim_dir, target, target + '.mat')
        default_out_dir = os.path.join(exp_dir, target, target + '_deepinverse') + '/'
    else:
        vis_path = os.path.join(base_dir, 'data','archival',target, target + '.mat')
        default_out_dir = os.path.join(base_dir, 'experiments', 'archival', target, target + '_deepinverse') + '/'
    
    out_dir = cfg.pop('out_dir', default_out_dir)
    if not out_dir.endswith('/'):
        out_dir += '/'
    real_projection = cfg["real_projection"]
    prior_name = cfg["prior"]
    prior_level = cfg["prior_level"]
    prior_wavelets = cfg["prior_wavelets"]
    prior_p = cfg["prior_p"]
    prior_clamp_min = cfg["prior_clamp_min"]
    stepsize_factor = cfg["stepsize_factor"]
    lambda_factor = cfg["lambda_factor"]
    params_algo_a = cfg["params_algo_a"]
    max_iter = cfg["max_iter"]
    early_stop = cfg["early_stop"]
    thres_conv = cfg["thres_conv"]
    img_size = tuple(cfg["img_size"])

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
        # if not name:
        #     from src.utils.transforms import normalize_fits_peak
        #     normalize_fits_peak(filepath)
        
        print(f"Saved: {filepath}")
        

    # Clear previous results
    import glob
    for f in glob.glob(os.path.join(out_dir, '*')):
        os.remove(f)

    # Load device
    device = dinv.utils.get_device()

    # Convert mat to npy
    mattonpy(vis_path, n=n)
    
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
        img_size=img_size,
        samples_loc=uv.permute((1, 0)),
        real_projection=real_projection,
        device=device,
    )
    # Add image weighting to the sensing operator
    physics.setWeight(nWimag)
    
    # Central Dirac pulse for PSF
    mock = torch.randint(5, (1, 1, img_size[0], img_size[1]))
    dirac = dirac_like(mock).to(device)
    
    # Compute PSF
    PSF = physics.A_adjoint(physics.A(dirac))
    savefits(PSF, 'psf')
   
    # Compute dirty image (single W via operator adjoint: Φᴴ · W · y_raw)
    x_hat = physics.A_adjoint(y.view(1, 1, -1).to(device))
    savefits(x_hat, 'dirty')
        
    # Initialization for FISTA: use pseudo-inverse (A_dagger) for better starting point
    x_init_fista = physics.A_dagger(y).to(device)
        
    # Operator norm cache path lives next to the input .mat file.
    opnorm_cache_path = vis_path.replace('.mat', '_opnorm.json')
    cached_opnorm = cfg.get("opnorm")

    if cached_opnorm is not None:
        opnorm = float(cached_opnorm)
        print(f"Using cached operator norm: {opnorm}")
    else:
        # Compute operator norm exactly as in DeepInv tutorial
        opnorm = physics.compute_sqnorm(
            torch.randn((1, 1, img_size[0], img_size[1]), device=device),
            max_iter=100,
            tol=1e-6,
            verbose=False,
        ).item()
        print(f"Operator norm (raw): {opnorm}")
        # Save it next to the .mat so later configs can reuse it.
        try:
            with open(opnorm_cache_path, 'w') as f:
                json.dump({"opnorm": opnorm}, f)
            print(f"Cached operator norm to {opnorm_cache_path}")
        except Exception as e:
            print(f"Warning: could not cache operator norm: {e}")
    
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
    from deepinv.optim import FISTA

    # Select the data fidelity term
    data_fidelity = L2()
    # Specify the prior (we redefine it with a smaller number of iteration for faster computation)
    wv_list = prior_wavelets
    prior = WaveletPrior(level=prior_level, wv=wv_list, p=prior_p, device=device, clamp_min=prior_clamp_min)

    # Logging parameters
    verbose = True
    plot_convergence_metrics = False # Log performance metrics

    # Algo parameters (aligned with demo_ri_basic.py)
    stepsize = 1.0 / (stepsize_factor * opnorm)
    lambda_reg = lambda_factor * opnorm  # Regularization parameter

    # Increase max_iter to allow FISTA to fully converge on the sharp edges
    max_iter = max_iter
    early_stop = early_stop

    def custom_init(y, physics):
        x_init = torch.clamp(physics.A_dagger(y), 0)
        return x_init

    model = FISTA(
        prior=prior,
        data_fidelity=data_fidelity,
        stepsize=stepsize,
        lambda_reg=lambda_reg,
        early_stop=early_stop,
        max_iter=max_iter,
        thres_conv=thres_conv,
        verbose=verbose,
        custom_init=custom_init,
    )

    init = torch.clamp(x_init_fista, 0), torch.clamp(x_init_fista, 0) # FISTA initialization (reuse dirty image)

    # Optional warm-up run. Skipped when a global warm-up has already been performed.
    if not cfg.get("skip_warmup", False):
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

    runtime = end_time - start_time
    imaging_time = end_time - start_time
    
    # PSF peak normalization is intentionally disabled to match the demo_ri_basic.py
    # workflow, which returns the FISTA solution in the input image scale without
    # rescaling by the PSF maximum.
    # psf_max = PSF.max().item()
    # if psf_max > 0:
    #     x_model = x_model / psf_max
    # print(f"PSF max: {psf_max:.6f} — reconstruction normalized by PSF max")

    # Flux-conserving calibration: scale the reconstruction so its total flux
    # matches the estimated total flux from the maximum visibility amplitude
    # (near-zero-spacing visibility ≈ total flux for compact/centred sources).
    total_flux_estimate = y_raw.abs().max().item()
    recon_sum = torch.clamp(x_model, min=0).sum().item()
    if recon_sum > 0 and total_flux_estimate > 0:
        flux_scale = total_flux_estimate / recon_sum
        x_model = x_model * flux_scale
        print(f"Flux calibration: estimated total flux = {total_flux_estimate:.4f}, "
              f"recon sum before = {recon_sum:.4f}, scale = {flux_scale:.6f}")

    savefits(x_model, '')

    # Log configuration and execution time
    log_filepath = os.path.join(out_dir, f"{target}_deepinverse.log")
    with open(log_filepath, "w") as f:
        f.write("DeepInv Configuration:\n")
        f.write("----------------------\n")
        f.write(f"Operator: RadioInterferometry\n")
        f.write(f"Image size: {img_size}\n")
        f.write(f"Real projection: {real_projection}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Operator norm (raw): {opnorm}\n")
        f.write(f"Flux estimate (max_visibility): {total_flux_estimate:.6f}\n")
        f.write(f"Flux calibration scale (flux_estimate / recon_sum): {flux_scale:.6e}\n\n")
        # f.write(f"Normalization scale: {scale} (1/sqrt(opnorm))\n\n")
        
        f.write("Optimization / Prior:\n")
        f.write("---------------------\n")
        f.write(f"Iteration: FISTA\n")
        f.write(f"Data Fidelity: L2\n")
        f.write(f"Prior: WaveletPrior (level={prior_level}, wv={wv_list}, p={prior_p}, clamp_min={prior_clamp_min})\n")
        f.write(f"Max iterations: {max_iter}\n")
        f.write(f"Early stop: {early_stop}\n")
        f.write(f"Convergence threshold: {thres_conv}\n")
        f.write(f"Stepsize: {stepsize} (1.0 / ({stepsize_factor} * opnorm))\n")
        f.write(f"Lambda: {lambda_reg} ({lambda_factor} * opnorm)\n")
        f.write(f"PSF peak normalization: disabled\n\n")
        
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


def mattonpy(mat, n=3):
    mat_data = scipy.io.loadmat(mat)
    vis_path = mat.replace('.mat', '.MS')
    y = mat_data.get('DATA', mat_data.get('y'))
    u = mat_data.get('u')
    v = mat_data.get('v')
    
    pixel_size = float(mat_data.get('nominal_pixelsize').flatten()[0])  # arcsec
    print(f'Pixel size (from .mat): {pixel_size} arcsec')
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
    parser.add_argument('--config', type=str, default=None, help='JSON string with DeepInverse parameters to override defaults')
    parser.add_argument('--nval', type=int, default=1, help='Factor for pixel size calculation')
    args = parser.parse_args()
    
    main(args.target, config_json=args.config, n=args.nval)