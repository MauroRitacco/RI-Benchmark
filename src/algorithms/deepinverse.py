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
        hdu.writeto(filepath, overwrite=True)
        print(f"Saved: {filepath}")
        

    # Load device
    device = dinv.utils.get_device()

    # Convert mat to npy
    mattonpy(vis_path)
    
    # Load npy. Important to specify dtype, otherwise it loads as float32
    data = dinv.utils.load_np(vis_path.replace('.mat', '.npy'), dtype=np.complex64, device=device)
    uv = dinv.utils.load_np(vis_path.replace('.mat', '_uv.npy'), dtype=np.float32, device=device)
    nWimag = dinv.utils.load_np(vis_path.replace('.mat', '_weight.npy'), dtype=np.float32, device=device)
    
    print(f"Data: {data.dtype} {data.shape}")
    print(f"UV: {uv.dtype} {uv.shape}")
    print(f"Weights: {nWimag.dtype} {nWimag.shape}")

    # Shape to (1, 1, N_vis)
    y = data.view(1, 1, -1).to(device)
   
    # Measurement operator
    physics = RadioInterferometry(
        (64, 64),
        samples_loc=uv.permute((1, 0)),
        real_projection=True,
        device=device,
    )

    # Weights reshape
    nWimag = nWimag.view(1, 1, -1)
    physics.setWeight(nWimag)
    
    # Central Dirac pulse for PSF
    mock = torch.randint(5, (1, 1, 64, 64))
    dirac = dirac_like(mock).to(device)
    
    # Compute PSF
    PSF = physics.A_adjoint(physics.A(dirac))
    savefits(PSF, 'psf')
   
    # Compute dirty image
    x_hat = physics.A_adjoint(y).to(device)
    savefits(x_hat, 'dirty')
        
    # Compute operator norm
    opnorm = physics.compute_sqnorm(
        torch.randn_like(x_hat, device=device),
        max_iter=20,
        tol=1e-6,
        verbose=False,
    ).item()
    print(f"Operator norm: {opnorm}")

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
    lamb = 1e-5 * opnorm  # Regularization parameter
    params_algo = {"stepsize": stepsize, "lambda": lamb, "a": 3}
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
    
    # Runtime measurement
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()

    # FISTA reconstruction
    init = torch.clamp(physics.A_dagger(y), 0), torch.clamp(physics.A_dagger(y), 0) # FISTA initialization
    x_model = model(y, physics, init=init)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    runtime = end_time - start_time
    
    # Log execution time
    with open(os.path.join(out_dir, f"{target}_deepinverse.log"), "w") as f:
        f.write(f"Execution time: {runtime:.2f} seconds\n")
    print(f"Execution time: {runtime:.6f} seconds")
    
    savefits(x_model, '')

    # Cleanup
    npy_file = vis_path.replace('.mat', '.npy')
    uv_npy_file = vis_path.replace('.mat', '_uv.npy')
    weight_npy_file = vis_path.replace('.mat', '_weight.npy')
    if os.path.exists(npy_file): os.remove(npy_file)
    if os.path.exists(uv_npy_file): os.remove(uv_npy_file)
    if os.path.exists(weight_npy_file): os.remove(weight_npy_file)


def normalize_uv(uv):
    """Normalize UV coordinates to [-pi, pi]."""
    max_val = np.abs(uv).max()
    if max_val > 0:
        uv = (uv / max_val) * np.pi
    return uv


def mattonpy(mat):
    mat_data = scipy.io.loadmat(mat)
    y = mat_data.get('DATA', mat_data.get('y'))
    u = mat_data.get('u')
    v = mat_data.get('v')
    
    # Scaling to match pixel scale (0.844 arcsec)
    cell_size_rad = 0.844 * np.pi / (180.0 * 3600.0)
    u = u * (2.0 * np.pi * cell_size_rad)
    v = v * (2.0 * np.pi * cell_size_rad)
    # max_val = np.abs(uv).max()
    # if max_val > 0:
    #     uv = (uv / max_val) * np.pi
    uv = np.concatenate((-v, u), axis=1)
    # uv = normalize_uv(uv)
    
    # Save arrays
    np.save(mat.replace('.mat', '.npy'), y)
    np.save(mat.replace('.mat', '_uv.npy'), uv)
    
    weight = mat_data.get('weight')
    if weight is not None:
        np.save(mat.replace('.mat', '_weight.npy'), weight)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MAT to NPY.')
    parser.add_argument('--target', type=str, required=True, help='Target target name')
    args = parser.parse_args()
    
    main(args.target)