import numpy as np
import os
import sys
from casatools import table
from astropy.io import fits
from casatasks import imsmooth

def normalize_fits_peak(fits_path):
    """Normalize a FITS image to [0, 1] via peak normalization; overwrite in place."""
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
    
    # Longest projected baseline (u, v) in meters
    tb.open(ms_path)
    uvw = tb.getcol('UVW')
    tb.close()
    longest_baseline_meters = np.max(np.sqrt(uvw[0]**2 + uvw[1]**2))

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
    factor = (4 * np.log(2) * hdr['CDELT1']**2) / (np.pi * hdr['BMAJ'] * hdr['BMIN'])
    data = data * factor
    if 'BUNIT' in hdr:
        hdr['BUNIT'] = 'Jy/pixel'
    fits.writeto(fits_path, data, hdr, overwrite=True)

def convert_jypixel_to_jybeam(fits_path):
    """Convert a FITS image from Jy/pixel to Jy/beam."""
    data, hdr = fits.getdata(fits_path, header=True)
    if 'BMAJ' not in hdr or 'BMIN' not in hdr:
        return  # can't convert without beam params
    factor = (np.pi * hdr['BMAJ'] * hdr['BMIN']) / (4 * np.log(2) * hdr['CDELT1']**2)
    data = data * factor
    if 'BUNIT' in hdr:
        hdr['BUNIT'] = 'Jy/beam'
    fits.writeto(fits_path, data, hdr, overwrite=True)

def smooth_to_beam(input_path, output_path, bmaj_deg, bmin_deg, bpa_deg):
    """Smooth an image to the target beam via CASA imsmooth."""
    # Delete existing output to avoid CASA collisions
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
    
    # imsmooth writes a CASA .image dir; export to FITS afterwards
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
    
    if os.path.exists(temp_casa_image):
        import shutil
        shutil.rmtree(temp_casa_image, ignore_errors=True)
    
    return output_path
    
def _get_mat_pixel_size(mat_path):
    """Return the nominal pixel size in arcsec from a .mat file, if available."""
    if not os.path.isfile(mat_path):
        return None
    try:
        from scipy.io import loadmat
        mat_data = loadmat(mat_path)
        if 'nominal_pixelsize' in mat_data:
            return float(np.squeeze(mat_data['nominal_pixelsize']))
    except Exception:
        pass
    try:
        import h5py
        with h5py.File(mat_path, 'r') as h5f:
            if 'nominal_pixelsize' in h5f:
                return float(np.squeeze(np.array(h5f['nominal_pixelsize'])))
    except Exception:
        pass
    return None


def formatfits(fits_path, ms_path, beam_params=None):
    """Write minimal WCS headers so CASA can read the FITS file (overwrite in place).

    Pixel size comes from ``nominal_pixelsize`` in the companion .mat file if
    present, else from the MS. Optionally adds beam params (arcsec/deg).
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

    # Pixel size: .mat nominal_pixelsize (same scale as R2D2/DeepInverse), else MS estimate
    mat_path = ms_path.rstrip(os.sep)
    if mat_path.lower().endswith('.ms'):
        mat_path = mat_path[:-3] + '.mat'
    else:
        mat_path = mat_path + '.mat'
    cell_arcsec = _get_mat_pixel_size(mat_path)
    if cell_arcsec is not None:
        print(f"  Using nominal_pixelsize from {mat_path}: {cell_arcsec:.6f} arcsec")
    else:
        cell_arcsec = calculate_pixel_size(ms_path)
    cell_deg = cell_arcsec / 3600.0

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

    # beam params in deg per FITS convention
    if beam_params is not None:
        hdr['BMAJ'] = beam_params['bmaj'] / 3600.0  # arcsec -> deg
        hdr['BMIN'] = beam_params['bmin'] / 3600.0  # arcsec -> deg
        hdr['BPA'] = beam_params['bpa']              # already in deg

    fits.writeto(fits_path, data_2d, hdr, overwrite=True)

def calculate_beam(psf_path, ms_path, box_radius=3):
    """Fit a 2D Gaussian to the PSF to get beam params (arcsec, arcsec, deg).

    Pixel scale derived from the MS. Avoids CASA PSF format issues.
    """
    from astropy.modeling import models, fitting

    data = fits.getdata(psf_path)
    data_2d = data.squeeze().astype(np.float64)
    if data_2d.ndim != 2:
        raise ValueError(f"Expected 2D PSF data after squeeze, got {data_2d.ndim}D")

    ny, nx = data_2d.shape
    y_center, x_center = ny // 2, nx // 2

    # Central box for fitting
    y_lo = max(y_center - box_radius, 0)
    y_hi = min(y_center + box_radius, ny)
    x_lo = max(x_center - box_radius, 0)
    x_hi = min(x_center + box_radius, nx)
    cutout = data_2d[y_lo:y_hi, x_lo:x_hi]

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

    bmaj_pix = max(fwhm_x, fwhm_y)
    bmin_pix = min(fwhm_x, fwhm_y)

    # astropy theta is CCW from +x; BPA is measured from North (+y) through East (-x),
    # i.e. from 90 deg in astropy coords towards 180 deg, so BPA = theta - 90 deg.
    theta_deg = np.degrees(gauss_fit.theta.value)

    # astropy theta tracks the x_stddev axis; if the major axis is along y,
    # its true PA is shifted by 90 deg
    if fwhm_y > fwhm_x:
        theta_deg += 90.0
        
    bpa_deg = theta_deg - 90.0
    
    # Normalize to [-90, 90] for standard BPA representation
    bpa_deg = ((bpa_deg + 90.0) % 180.0) - 90.0

    # pixels -> arcsec
    cell_arcsec = calculate_pixel_size(ms_path)
    bmaj_arcsec = bmaj_pix * cell_arcsec
    bmin_arcsec = bmin_pix * cell_arcsec

    print(f"  Fitted Gaussian: bmaj={bmaj_pix:.2f} px, bmin={bmin_pix:.2f} px, theta={theta_deg:.1f} deg")

    return {'bmaj': bmaj_arcsec, 'bmin': bmin_arcsec, 'bpa': bpa_deg}