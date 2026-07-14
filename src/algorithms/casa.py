import sys
import os
# Add project root directory to python path for top-level module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import argparse
import json
import casatasks
import time
import numpy as np
from astropy.io import fits
from src.utils.transforms import convert_jybeam_to_jypixel
# from src.utils.transforms import smoothimage

def main(target, config_json=None, n=3):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')) # Get the project root directory (2 levels up from this script)
    # Check if the target is simulated or archival
    if os.path.exists(os.path.join(base_dir,'data','simulated',target)):
        vis_path = os.path.join(base_dir, 'data','simulated',target, target + '.MS') #Path to the MS file
        out_dir = os.path.join(base_dir, 'experiments', 'simulated', target, target + '_casa') + '/' #Output directory for the experiment
    else:
        vis_path = os.path.join(base_dir, 'data','archival',target, target + '.MS') #Path to the MS file
        out_dir = os.path.join(base_dir, 'experiments', 'archival', target, target + '_casa') + '/' #Output directory for the experiment

    name = out_dir + target + '_casa' #Prefix for all generated images
    # Delete the output directory if it exists to start completely fresh
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    
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
    # Load the nominal_pixelsize from the .mat file — this is the exact pixel scale used
    # during visibility simulation, so CASA images at the same angular scale as the groundtruth.
    from scipy.io import loadmat
    mat_path = vis_path.replace('.MS', '.mat')
    pixel_size = float(loadmat(mat_path)['nominal_pixelsize'].item())
    print(f'Pixel size (from .mat): {pixel_size} arcsec')

    # Parse user-provided tclean config (JSON dict from shell script)
    user_config = {}
    if config_json:
        user_config = json.loads(config_json)
        print(f"User tclean config: {user_config}")

    t0_start = time.perf_counter()
    # Build dirty image kwargs: start from user config, then force the dirty-specific overrides
    dirty_kwargs = dict(user_config)
    dirty_kwargs.update({
        'vis': vis_path,
        'imagename': name,
        'cell': f'{pixel_size}arcsec',
        'niter': 0,
    })
    # Remove clean-only keys that don't apply to the dirty run
    for key in ['scales', 'gain', 'cyclefactor', 'usemask', 'nsigma', 'noisethreshold']:
        dirty_kwargs.pop(key, None)

    casatasks.tclean(**dirty_kwargs)
    t0_end = time.perf_counter()
    setup_time = t0_end - t0_start
    
    # Erase every output of the first tclean except .image
    import glob
    for f in glob.glob(f"{name}.*"):
        if f not in [f"{name}.image", f"{name}.log"]:
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)
                
    # Rename .image to .dirty
    if os.path.exists(f"{name}.image"):
        os.rename(f"{name}.image", f"{name}.dirty")
    
    start = time.perf_counter()

    # Use user config directly, then force computed values
    tclean_kwargs = dict(user_config)
    tclean_kwargs['vis'] = vis_path
    tclean_kwargs['imagename'] = name
    tclean_kwargs['cell'] = f'{pixel_size}arcsec'

    casatasks.tclean(**tclean_kwargs)
    
    # Stop timer
    end = time.perf_counter()
    duration = end - start
    imaging_time = duration - setup_time
    if imaging_time < 0:
        imaging_time = duration
    
    # Export the dirty image to a FITS file if it exists
    if os.path.exists(f'{name}.dirty'):
        casatasks.exportfits(
            imagename=f'{name}.dirty', 
            fitsimage=f'{name}_dirty.fits', 
            overwrite=True
        )
        
    # Export the clean image to a FITS file if it exists
    if os.path.exists(f'{name}.image'):
        casatasks.exportfits(
            imagename=f'{name}.image', 
            fitsimage=f'{name}.fits', 
            overwrite=True
        )
        # smoothimage(f'{name}.fits', os.path.join(base_dir,'data','simulated',target, target + '.fits'))
        
        # # Convert units to Jy/pixel
        convert_jybeam_to_jypixel(f'{name}.fits')

        # Normalize to [0, 1] via peak normalization
        # from src.utils.transforms import normalize_fits_peak
        # normalize_fits_peak(f'{name}.fits')
    
    # Write the execution time and tclean config to the log file
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