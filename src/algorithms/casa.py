import argparse
import casatasks
import os
import time

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
    os.makedirs(out_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    
    # Redirect CASA log to the experiment folder and remove the default one
    old_log = casatasks.casalog.logfile()
    casatasks.casalog.setlogfile(name+'.log')
    try:
        if os.path.exists(old_log):
            os.remove(old_log)
    except Exception:
        pass

    # Reset CASA statistics
    casatasks.clearstat() 

    # Start timer
    start = time.perf_counter()

    # Run tclean task
    # Remove existing files so tclean starts from scratch instead of resuming
    os.system(f"rm -rf {name}.*")
    casatasks.tclean(
        vis=vis_path, 
        imagename=name, 
        imsize=64, 
        cell='8.44e-1arcsec', 
        specmode='mfs',
        deconvolver='hogbom', 
        gridder='standard', 
        weighting='briggs',
        robust=0, 
        niter=10000,
        datacolumn='data'
    )
    
    # Stop timer
    end = time.perf_counter()
    duration = end - start
    
    # Export the image to a FITS file
    casatasks.exportfits(
        imagename=f'{name}.image', 
        fitsimage=f'{name}.fits', 
        overwrite=True
    )
    
    # Write the execution time to the log file
    with open(name+'.log', 'a') as f:
        f.write(f"\nExecution time: {duration:.2f} seconds\n")
    print(f"Execution time: {duration} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CASA Hogbom clean on a target.")
    parser.add_argument("--target", required=True, help="Name of the object to image")
    args = parser.parse_args()
    main(args.target)