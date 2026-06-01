import sys
import os
# Add project root directory to python path for top-level module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import argparse
import casatasks
import time
import numpy as np
from astropy.io import fits
from src.utils.transforms import convert_jybeam_to_jypixel
from src.utils.transforms import smoothimage

def main(target):
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
    from src.utils.transforms import calculate_pixel_size
    pixel_size = calculate_pixel_size(vis_path,n=2)
    print(f'Pixel size: {pixel_size} arcsec')

    # Run tclean task
    casatasks.tclean(
        vis=vis_path, 
        imagename=name, 
        imsize=64, 
        cell=f'{pixel_size}arcsec', 
        specmode='mfs',
        deconvolver='hogbom', 
        gridder='standard', 
        weighting='briggs',
        robust=0,
        niter=0,
        datacolumn='data'
    )
    
    # Erase every output of the first tclean except .image
    import glob
    for f in glob.glob(f"{name}.*"):
        if f not in [f"{name}.image"]:
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)
                
    # Rename .image to .dirty
    if os.path.exists(f"{name}.image"):
        os.rename(f"{name}.image", f"{name}.dirty")
    
    start = time.perf_counter()
    tclean_kwargs = {
        'vis': vis_path, 
        'imagename': name, 
        'imsize': 64, 
        'cell': f'{pixel_size}arcsec', 
        'specmode': 'mfs',
        'deconvolver': 'hogbom', 
        'gridder': 'standard', 
        'weighting': 'briggs',
        'robust': 0,
        'gain': 0.1,
        'niter': 1000000,
        'threshold': '0.001Jy',
        'datacolumn': 'data'
    }
    casatasks.tclean(**tclean_kwargs)
    
    # Stop timer
    end = time.perf_counter()
    duration = end - start
    
    # Export the image to a FITS file
    casatasks.exportfits(
        imagename=f'{name}.image', 
        fitsimage=f'{name}.fits', 
        overwrite=True
    )
    smoothimage(f'{name}.fits', os.path.join(base_dir,'data','simulated',target, target + '.fits'))
    
    # Convert units to Jy/pixel
    convert_jybeam_to_jypixel(f'{name}.fits')
    
    # Write the execution time and tclean config to the log file
    with open(name+'.log', 'a') as f:
        f.write(f"tclean config: {tclean_kwargs}\n")
        f.write(f"\nExecution time: {duration:.2f} seconds\n")
    print(f"Execution time: {duration:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CASA Hogbom clean on a target.")
    parser.add_argument("--target", required=True, help="Name of the object to image")
    args = parser.parse_args()
    main(args.target)