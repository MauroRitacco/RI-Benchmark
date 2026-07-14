import numpy as np
import os
import sys
from casatools import table
from astropy.io import fits
from casatasks import imsmooth

def normalize_fits_peak(fits_path):
    """Normalize a FITS image to [0, 1] via peak normalization and clip negatives.
    
    Divides all pixel values by the maximum and clips negatives to zero,
    so the output range is [0, 1]. Overwrites the file in place.
    """
    data, hdr = fits.getdata(fits_path, header=True)
    data = data.astype(np.float64)
    dmax = data.max()
    if dmax != 0:
        data = np.clip(data / dmax, 0, None)
    else:
        data = np.zeros_like(data)
    fits.writeto(fits_path, data.astype(np.float32), hdr, overwrite=True)

def calculate_pixel_size(ms_path, n=2.0):
    tb = table()
    
    # Get the longest projected baseline in meters (u, v)
    tb.open(ms_path)
    uvw = tb.getcol('UVW')
    tb.close()
    longest_baseline_meters = np.max(np.sqrt(uvw[0]**2 + uvw[1]**2))
    
    # Get the reference frequency and calculate wavelength
    tb.open(ms_path + '/SPECTRAL_WINDOW')
    try:
        frequency = tb.getcol('CHAN_FREQ')[0, 0]
    except RuntimeError:
        frequency = tb.getcell('CHAN_FREQ', 0)[0]
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

def convert_jypixel_to_jybeam(fits_path):
    """Convert a FITS image from Jy/pixel to Jy/beam."""
    data, hdr = fits.getdata(fits_path, header=True)
    if 'BMAJ' not in hdr or 'BMIN' not in hdr:
        return # Can't convert without beam params
    # Unit conversion: Jy/pixel -> Jy/beam
    factor = (np.pi * hdr['BMAJ'] * hdr['BMIN']) / (4 * np.log(2) * hdr['CDELT1']**2)
    data = data * factor
    if 'BUNIT' in hdr:
        hdr['BUNIT'] = 'Jy/beam'
    fits.writeto(fits_path, data, hdr, overwrite=True)

def smooth_to_beam(input_path, output_path, bmaj_deg, bmin_deg, bpa_deg):
    """Smooth an image to a specified beam using CASA imsmooth.
    
    Parameters
    ----------
    input_path : str
        Path to the input FITS file to be smoothed.
    output_path : str
        Path where the smoothed FITS file will be saved.
    bmaj_deg : float
        Major axis of the target beam in degrees.
    bmin_deg : float
        Minor axis of the target beam in degrees.
    bpa_deg : float
        Position angle of the target beam in degrees.
    """
    # Delete output file if it exists to avoid CASA collision issues
    if os.path.exists(output_path):
        try:
            import shutil
            if os.path.isdir(output_path):
                shutil.rmtree(output_path)
            else:
                os.remove(output_path)
        except Exception:
            pass

    BMAJ = f"{bmaj_deg}deg"
    BMIN = f"{bmin_deg}deg"
    PA = f"{bpa_deg}deg"

    temp_dir = os.path.dirname(output_path)
    os.makedirs(temp_dir, exist_ok=True)
    
    # imsmooth outputs a CASA image directory. We write to a temporary .image first.
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    temp_casa_image = os.path.join(temp_dir, f"{base_name}_temp_smoothed.image")
    if os.path.exists(temp_casa_image):
        import shutil
        shutil.rmtree(temp_casa_image, ignore_errors=True)

    class SuppressOutput:
        def __enter__(self):
            self.null_fd = os.open(os.devnull, os.O_RDWR)
            self.save_stdout = os.dup(1)
            self.save_stderr = os.dup(2)
            os.dup2(self.null_fd, 1)
            os.dup2(self.null_fd, 2)
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            os.dup2(self.save_stdout, 1)
            os.dup2(self.save_stderr, 2)
            os.close(self.null_fd)
            os.close(self.save_stdout)
            os.close(self.save_stderr)

    try:
        with SuppressOutput():
            imsmooth(
                imagename=input_path,
                kernel="gaussian",
                beam={"major": BMAJ, "minor": BMIN, "pa": PA},
                targetres=True,
                outfile=temp_casa_image,
                overwrite=True,
            )
            from casatasks import exportfits
            exportfits(
                imagename=temp_casa_image,
                fitsimage=output_path,
                overwrite=True
            )
    except RuntimeError as e:
        error_msg = str(e)
        if "identical to or larger than" in error_msg or "setAxialRatio" in error_msg:
            # Beam is already at or above target resolution — no further smoothing needed.
            print(f"Note: Beam already at/above target resolution. Copying original.")
            import shutil
            shutil.copy2(input_path, output_path)
        else:
            raise e
    
    # Cleanup temporary files
    if os.path.exists(temp_casa_image):
        import shutil
        shutil.rmtree(temp_casa_image, ignore_errors=True)
    
    return output_path
    
def formatfits(fits_path, ms_path, beam_params=None):
    """Add proper WCS headers to a FITS file so CASA can read it.
    
    Derives image dimensions from the data itself and computes pixel size from the MS.
    Optionally adds beam parameters (BMAJ, BMIN, BPA) to the header.
    Overwrites the FITS file in-place with the new header.
    
    Parameters
    ----------
    fits_path : str
        Path to the FITS file to format.
    ms_path : str
        Path to the measurement set, used to derive pixel size.
    beam_params : dict, optional
        Dictionary with keys 'bmaj' (arcsec), 'bmin' (arcsec), 'bpa' (deg).
    """
    data = fits.getdata(fits_path)
    data_2d = data.squeeze()
    ny, nx = data_2d.shape

    hdr = fits.Header()
    hdr['SIMPLE'] = True
    hdr['BITPIX'] = -32
    hdr['NAXIS'] = 2
    hdr['NAXIS1'] = nx
    hdr['NAXIS2'] = ny
    hdr['BUNIT'] = 'Jy/pixel'

    # Compute cell size in degrees from the MS
    cell_deg = calculate_pixel_size(ms_path) / 3600.0

    # WCS coordinate keywords required by CASA
    hdr['CTYPE1'] = 'RA---SIN'
    hdr['CRVAL1'] = 0.0
    hdr['CDELT1'] = -cell_deg  # RA axis (negative by convention)
    hdr['CRPIX1'] = (nx / 2) + 0.5
    hdr['CUNIT1'] = 'deg'

    hdr['CTYPE2'] = 'DEC--SIN'
    hdr['CRVAL2'] = 0.0
    hdr['CDELT2'] = cell_deg
    hdr['CRPIX2'] = (ny / 2) + 0.5
    hdr['CUNIT2'] = 'deg'

    # Add beam parameters if provided (stored in degrees per FITS convention)
    if beam_params is not None:
        hdr['BMAJ'] = beam_params['bmaj'] / 3600.0  # arcsec -> deg
        hdr['BMIN'] = beam_params['bmin'] / 3600.0  # arcsec -> deg
        hdr['BPA'] = beam_params['bpa']              # already in deg

    fits.writeto(fits_path, data_2d, hdr, overwrite=True)

def calculate_beam(psf_path, ms_path, box_radius=3):
    """Calculate beam parameters (bmaj, bmin, bpa) by fitting a 2D Gaussian to the PSF.
    
    Uses astropy.modeling to fit the PSF directly, avoiding CASA format issues.
    
    Parameters
    ----------
    psf_path : str
        Path to the PSF FITS file.
    ms_path : str
        Path to the measurement set, used to derive pixel size.
    box_radius : int, optional
        Half-size of the bounding box around the PSF center for fitting (default: 20).
    
    Returns
    -------
    dict
        Dictionary with keys 'bmaj' (arcsec), 'bmin' (arcsec), 'bpa' (deg).
    """
    from astropy.modeling import models, fitting

    data = fits.getdata(psf_path)
    data_2d = data.squeeze().astype(np.float64)
    if data_2d.ndim != 2:
        raise ValueError(f"Expected 2D PSF data after squeeze, got {data_2d.ndim}D")

    ny, nx = data_2d.shape
    y_center, x_center = ny // 2, nx // 2

    # Extract the central box for fitting
    y_lo = max(y_center - box_radius, 0)
    y_hi = min(y_center + box_radius, ny)
    x_lo = max(x_center - box_radius, 0)
    x_hi = min(x_center + box_radius, nx)
    cutout = data_2d[y_lo:y_hi, x_lo:x_hi]

    # Create coordinate grids for the cutout
    y_grid, x_grid = np.mgrid[0:cutout.shape[0], 0:cutout.shape[1]]

    # Initial guess: peak at center of cutout, sigma ~3 pixels
    x_mean_init = cutout.shape[1] / 2.0
    y_mean_init = cutout.shape[0] / 2.0
    amplitude_init = cutout.max()

    gauss_init = models.Gaussian2D(
        amplitude=amplitude_init,
        x_mean=x_mean_init,
        y_mean=y_mean_init,
        x_stddev=3.0,
        y_stddev=3.0,
        theta=0.0,
    )

    fitter = fitting.LevMarLSQFitter()
    gauss_fit = fitter(gauss_init, x_grid, y_grid, cutout)

    # Convert sigma to FWHM (FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.3548 * sigma)
    fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
    fwhm_x = abs(gauss_fit.x_stddev.value) * fwhm_factor  # pixels
    fwhm_y = abs(gauss_fit.y_stddev.value) * fwhm_factor  # pixels

    # bmaj = larger FWHM, bmin = smaller FWHM
    bmaj_pix = max(fwhm_x, fwhm_y)
    bmin_pix = min(fwhm_x, fwhm_y)

    # Position angle: astropy theta is measured counter-clockwise from +x axis.
    # In radio astronomy FITS images, North is +y and East is -x.
    # BPA is measured from North through East (counter-clockwise).
    # Therefore, the North axis (+y) corresponds to 90 deg in astropy, 
    # and we want to measure from there towards East (-x, which is 180 deg in astropy).
    # So BPA = theta_astropy - 90 deg.
    theta_deg = np.degrees(gauss_fit.theta.value)
    
    # If major axis is along y (fwhm_y > fwhm_x), the true position angle of the major axis
    # is shifted by 90 degrees compared to astropy's theta (which tracks the x_stddev axis).
    if fwhm_y > fwhm_x:
        theta_deg += 90.0
        
    bpa_deg = theta_deg - 90.0
    
    # Normalize to [-90, 90] for standard BPA representation
    bpa_deg = ((bpa_deg + 90.0) % 180.0) - 90.0

    # Convert from pixels to arcsec using the pixel scale from the MS
    cell_arcsec = calculate_pixel_size(ms_path)
    bmaj_arcsec = bmaj_pix * cell_arcsec
    bmin_arcsec = bmin_pix * cell_arcsec

    print(f"  Fitted Gaussian: bmaj={bmaj_pix:.2f} px, bmin={bmin_pix:.2f} px, theta={theta_deg:.1f} deg")

    return {'bmaj': bmaj_arcsec, 'bmin': bmin_arcsec, 'bpa': bpa_deg}