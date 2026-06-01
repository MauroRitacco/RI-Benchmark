import numpy as np
import os
from casatools import table
from astropy.io import fits
from casatasks import imsmooth

def calculate_pixel_size(ms_path, n=2.0):
    tb = table()
    
    # Get the longest projected baseline in meters (u, v)
    tb.open(ms_path)
    uvw = tb.getcol('UVW')
    tb.close()
    longest_baseline_meters = np.max(np.sqrt(uvw[0]**2 + uvw[1]**2))
    
    # Get the reference frequency and calculate wavelength
    tb.open(ms_path + '/SPECTRAL_WINDOW')
    frequency = tb.getcol('CHAN_FREQ')[0, 0]
    tb.close()
    
    wavelength = 299792458.0 / frequency
    longest_baseline_lambda = longest_baseline_meters / wavelength
    
    # cell(arcsec) = (180 * 3600 / pi) * (1 / (superresolution * 2 * maxProjBaseline))
    return (180.0 * 3600.0 / np.pi) * (1.0 / (n * longest_baseline_lambda))

def convert_jybeam_to_jypixel(fits_path):
    """Convert a FITS image from Jy/beam to Jy/pixel."""
    data, hdr = fits.getdata(fits_path, header=True)
    # Unit conversion: Jy/beam -> Jy/pixel
    factor = (4 * np.log(2) * hdr['CDELT1']**2) / (np.pi * hdr['BMAJ'] * hdr['BMIN'])
    data = data * factor
    if 'BUNIT' in hdr:
        hdr['BUNIT'] = 'Jy/pixel'
    fits.writeto(fits_path, data, hdr, overwrite=True)

def smoothimage(reconstruction_path, ground_truth_path):
    recon_hdr = fits.getheader(reconstruction_path)
    
    # if 'BMAJ' not in recon_hdr or 'BMIN' not in recon_hdr:
    #     return ground_truth_path
        
    BMAJ=str(recon_hdr['BMAJ']) + 'deg'
    BMIN=str(recon_hdr['BMIN']) + 'deg'
    PA=str(recon_hdr.get('BPA', 0)) + 'deg'

    # Target experiment directory (e.g. experiments/simulated/disk_0000)
    experiments_target_dir = os.path.dirname(os.path.dirname(reconstruction_path))
    os.makedirs(experiments_target_dir, exist_ok=True)
    target_name = os.path.splitext(os.path.basename(ground_truth_path))[0]

    # Delete output file if it exists to avoid CASA collision issues
    out_file = os.path.join(experiments_target_dir, f"{target_name}_smoothed.fits")
    if os.path.exists(out_file):
        try:
            import shutil
            if os.path.isdir(out_file):
                shutil.rmtree(out_file)
            else:
                os.remove(out_file)
        except Exception:
            pass

    # If the ground truth FITS is missing proper WCS headers, fix it first
    gt_hdr = fits.getheader(ground_truth_path)
    if 'CTYPE1' not in gt_hdr:
        gt_dir = os.path.dirname(ground_truth_path)
        ms_path = os.path.join(gt_dir, f"{target_name}.MS")
        formatfits(ground_truth_path, ms_path)

    # imsmooth outputs a CASA image directory. We write to a temporary .image first.
    temp_casa_image = os.path.join(experiments_target_dir, f"{target_name}_temp_smoothed.image")
    if os.path.exists(temp_casa_image):
        import shutil
        shutil.rmtree(temp_casa_image, ignore_errors=True)

    imsmooth(
    imagename=ground_truth_path,
    kernel="gaussian",
    beam={"major": BMAJ, "minor": BMIN, "pa": PA},
    targetres=True,
    outfile=temp_casa_image,
    overwrite=True,
    )

    from casatasks import exportfits
    exportfits(
        imagename=temp_casa_image,
        fitsimage=out_file,
        overwrite=True
    )
    
    # Cleanup temporary files
    if os.path.exists(temp_casa_image):
        import shutil
        shutil.rmtree(temp_casa_image, ignore_errors=True)
    
    return out_file
    
def formatfits(fits_path, ms_path):
    imsize = 64 
    data = fits.getdata(fits_path)

    hdr = fits.Header()
    hdr['SIMPLE'] = True
    hdr['BITPIX'] = -32
    hdr['NAXIS'] = 2
    hdr['NAXIS1'] = imsize
    hdr['NAXIS2'] = imsize
    if not 'BUNIT' in hdr or hdr['BUNIT']!='Jy/beam':
        hdr['BUNIT']='Jy/beam'
    # Convertir cell a grados para el estándar WCS usando el ms_path
    cell_deg = calculate_pixel_size(ms_path) / 3600.0

    # WCS coordinate keywords required by CASA
    hdr['CTYPE1'] = 'RA---SIN'
    hdr['CRVAL1'] = 0.0
    hdr['CDELT1'] = -cell_deg  # Eje X (negativo por convención de RA)
    hdr['CRPIX1'] = (imsize / 2) + 0.5
    hdr['CUNIT1'] = 'deg'

    hdr['CTYPE2'] = 'DEC--SIN'
    hdr['CRVAL2'] = 0.0
    hdr['CDELT2'] = cell_deg   # Eje Y (DEC)
    hdr['CRPIX2'] = (imsize / 2) + 0.5
    hdr['CUNIT2'] = 'deg'

    # Guardar el FITS con WCS válido
    fits.writeto(fits_path, data, hdr, overwrite=True)
