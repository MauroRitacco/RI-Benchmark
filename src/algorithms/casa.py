import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import argparse
import json
import casatasks
import time
import numpy as np
from astropy.io import fits
from src.utils.transforms import convert_jybeam_to_jypixel
from src.utils.dirty_noise import estimate_noise, log_noise_result
# from src.utils.transforms import smoothimage


def _parse_auto_threshold(threshold):
    """Return (is_auto, multiplier) for threshold values like 'auto:3' or 'auto:1.5'."""
    if isinstance(threshold, str) and threshold.startswith("auto:"):
        try:
            return True, float(threshold.split(":", 1)[1])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid auto threshold format: {threshold}") from exc
    return False, None


def _find_gt_path(base_dir, target):
    """Return the ground-truth FITS path if it exists, else None."""
    if os.environ.get("RI_BENCH_TEST") == "1":
        gt = os.path.join(base_dir, "data", "simulated", "test", target, f"{target}.fits")
        return gt if os.path.exists(gt) else None
    for category in ("simulated", "archival"):
        gt = os.path.join(base_dir, "data", category, target, f"{target}.fits")
        if os.path.exists(gt):
            return gt
    return None


def _append_noise_log(log_path, noise_result, multiplier, threshold):
    """Append a structured noise-estimation block to the experiment log."""
    import logging
    logger = logging.getLogger("casa_noise_log")
    logger.handlers = []
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    logger.info("[noise] Auto threshold resolved from dirty image")
    log_noise_result(noise_result, log_func=logger.info)
    logger.info("[noise] multiplier=%.2f -> threshold=%.6e Jy", multiplier, threshold)


def main(target, config_json=None, n=3):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    # Test mode reads from data/simulated/test/
    if os.environ.get("RI_BENCH_TEST") == "1":
        sim_dir = os.path.join(base_dir, 'data', 'simulated', 'test')
        exp_dir = os.path.join(base_dir, 'experiments', 'test')
    else:
        sim_dir = os.path.join(base_dir, 'data', 'simulated')
        exp_dir = os.path.join(base_dir, 'experiments', 'simulated')
    if os.path.exists(os.path.join(sim_dir, target)):
        vis_path = os.path.join(sim_dir, target, target + '.MS')
        default_out_dir = os.path.join(exp_dir, target, target + '_casa') + '/'
    else:
        vis_path = os.path.join(base_dir, 'data','archival',target, target + '.MS')
        default_out_dir = os.path.join(base_dir, 'experiments', 'archival', target, target + '_casa') + '/'

    # Parse user-provided tclean config (JSON dict from shell script)
    user_config = {}
    if config_json:
        user_config = json.loads(config_json)
        print(f"User tclean config: {user_config}")

    # Allow overriding the output prefix so the sweep driver can place each config in its own subdir
    if 'imagename' in user_config:
        name = user_config.pop('imagename')
        out_dir = os.path.dirname(name) + '/'
    else:
        name = default_out_dir + target + '_casa'
        out_dir = default_out_dir
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Redirect CASA log to the experiment folder and remove the default one
    old_log = casatasks.casalog.logfile()
    casatasks.casalog.setlogfile(name+'.log')
    try:
        if os.path.exists(old_log):
            os.remove(old_log)
    except Exception:
        pass

    import sys
    sys.path.insert(0, base_dir)
    # nominal_pixelsize from the .mat: exact pixel scale used during visibility simulation,
    # so CASA images at the same angular scale as the groundtruth.
    from scipy.io import loadmat
    mat_path = vis_path.replace('.MS', '.mat')
    pixel_size = float(loadmat(mat_path)['nominal_pixelsize'].item())
    print(f'Pixel size (from .mat): {pixel_size} arcsec')

    t0_start = time.perf_counter()
    dirty_kwargs = dict(user_config)
    dirty_kwargs.update({
        'vis': vis_path,
        'imagename': name,
        'cell': f'{pixel_size}arcsec',
        'niter': 0,
    })
    # Remove clean-only keys; an auto threshold is recomputed below from the dirty image
    for key in ['scales', 'gain', 'cyclefactor', 'usemask', 'nsigma', 'noisethreshold', 'threshold']:
        dirty_kwargs.pop(key, None)

    casatasks.tclean(**dirty_kwargs)
    t0_end = time.perf_counter()
    setup_time = t0_end - t0_start

    # Keep only .image/.log from the dirty pass
    import glob
    for f in glob.glob(f"{name}.*"):
        if f not in [f"{name}.image", f"{name}.log"]:
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)

    if os.path.exists(f"{name}.image"):
        os.rename(f"{name}.image", f"{name}.dirty")
    dirty_fits = f"{name}_dirty.fits"
    if os.path.exists(f'{name}.dirty'):
        casatasks.exportfits(
            imagename=f'{name}.dirty',
            fitsimage=dirty_fits,
            overwrite=True
        )

    tclean_kwargs = dict(user_config)
    is_auto, multiplier = _parse_auto_threshold(tclean_kwargs.get('threshold'))
    if is_auto:
        gt_path = _find_gt_path(base_dir, target)
        noise_result = estimate_noise(
            dirty_path=dirty_fits,
            gt_path=gt_path,
            multiplier=multiplier,
        )
        threshold_jy = noise_result.threshold
        tclean_kwargs['threshold'] = f'{threshold_jy}Jy'
        casatasks.casalog.post(
            f"[noise] auto threshold: multiplier={multiplier}, sigma_noise={noise_result.sigma_noise:.6e}, "
            f"threshold={threshold_jy:.6e} Jy", 'INFO'
        )
        _append_noise_log(name + '.log', noise_result, multiplier, threshold_jy)

    start = time.perf_counter()

    tclean_kwargs['vis'] = vis_path
    tclean_kwargs['imagename'] = name
    tclean_kwargs['cell'] = f'{pixel_size}arcsec'

    casatasks.tclean(**tclean_kwargs)

    end = time.perf_counter()
    duration = end - start
    imaging_time = duration - setup_time
    if imaging_time < 0:
        imaging_time = duration
    
    if os.path.exists(f'{name}.image'):
        casatasks.exportfits(
            imagename=f'{name}.image', 
            fitsimage=f'{name}.fits', 
            overwrite=True
        )
        # smoothimage(f'{name}.fits', os.path.join(base_dir,'data','simulated',target, target + '.fits'))

        convert_jybeam_to_jypixel(f'{name}.fits')

        # Normalize to [0, 1] via peak normalization
        # from src.utils.transforms import normalize_fits_peak
        # normalize_fits_peak(f'{name}.fits')
    
    with open(name+'.log', 'a') as f:
        f.write("tclean config:\n")
        f.write(json.dumps(tclean_kwargs, indent=2, default=str) + "\n")
        f.write(f"\nExecution time: {duration:.2f} seconds\n")
        f.write(f"Imaging time: {imaging_time:.2f} seconds\n")
    print(f"Execution time: {duration:.2f} seconds")
    print(f"Imaging time: {imaging_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CASA Hogbom clean on a target.")
    parser.add_argument("--target", required=True, help="Name of the object to image")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON string with tclean parameters to override defaults")
    parser.add_argument("--nval", type=int, default=3, help="Factor for pixel size calculation")
    args = parser.parse_args()
    main(args.target, config_json=args.config, n=args.nval)