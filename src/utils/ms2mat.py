FF# Extract data from MS into a .mat file
# Author: A. Dabbech
# Adapted from a script in the R2D2 repository (basp-group/R2D2-RI).
import os
import sys
import argparse
import numpy as np
import scipy.io as sio
import math
from casacore import tables

c = 299792458  # speed of light (m/s)


def main():
    parser = argparse.ArgumentParser(description='Extract data file from MS')
    parser.add_argument('--msfile', type=str, default=None, help='Path to MS file')
    parser.add_argument('--srcname', type=str, default='', help='Source name')
    parser.add_argument('--srcid', type=int, default=0, help='Source/Field ID')
    parser.add_argument('--freqid', type=int, default=0, help='Channel index (used when not --all_channels)')
    parser.add_argument('--all_channels', action='store_true', help='Export ALL channels of each SPW (frequency ends up embedded in the uv coordinates)')
    parser.add_argument('--average_channels', action='store_true', help='Collapse ALL channels of each SPW into ONE noise-weighted visibility per row (continuum mode; uv at the mean frequency of the valid channels)')
    parser.add_argument('--uv_cutoff', type=float, default=None, help='Max baseline in wavelengths')
    args = parser.parse_args()
    if args.all_channels and args.average_channels:
        raise SystemExit("--all_channels and --average_channels are mutually exclusive")
    msfile =  args.msfile
    srcname = args.srcname
    srcid = args.srcid
    freqid = args.freqid

    tab = tables.table(msfile)
    print("INFO: MS table columns:", *(tab.colnames()))
    print("INFO: Reading data ..Freq %s" % freqid)
    ## freq channels
    spwtab = tables.table("%s/SPECTRAL_WINDOW"%msfile)
    try:
        freqs = spwtab.getcol("CHAN_FREQ")
        frequency = freqs[0, freqid]
    except RuntimeError:
        freqs = spwtab.getcell("CHAN_FREQ", 0)
        frequency = freqs[freqid]
    spwtab.close()

    print("INFO: Extracting data from all Spectral Windows...")
    spw_ids = np.unique(tab.getcol("DATA_DESC_ID"))
    
    all_y, all_u, all_v, all_w, all_nW, all_nWimag = [], [], [], [], [], []
    
    for spw_id in spw_ids:
        subtab = tab.query("FIELD_ID == %s AND DATA_DESC_ID == %s" % (srcid, spw_id))
        nmeas = subtab.nrows()
        if nmeas == 0:
            continue
            
        spwtab = tables.table("%s/SPECTRAL_WINDOW" % msfile)
        try:
            freqs_spw = spwtab.getcell("CHAN_FREQ", spw_id)
        except Exception:
            freqs_spw = np.array([frequency])  # fallback to global frequency
        spwtab.close()
        nchan_spw = len(freqs_spw)

        try:
            data = subtab.getcol("CORRECTED_DATA")
        except:
            data = subtab.getcol("DATA")
        ncorr = data.shape[2]

        ## natural weights = 1/variance
        try:
            weight_ch = subtab.getcol("WEIGHT_SPECTRUM")  # (nmeas, nchan, ncorr)
        except:
            weight_ch = None
            weight = subtab.getcol("WEIGHT")  # (nmeas, ncorr), row-level fallback

        flag_row = subtab.getcol("FLAG_ROW")
        flag_row = (np.reshape(np.array(flag_row), (nmeas,))).astype(float)
        flag_cube = (subtab.getcol("FLAG")).astype(float)  # (nmeas, nchan, ncorr)

        uvw = subtab.getcol("UVW")

        ## briggs/uniform imaging weights
        weight_imaging = None
        try:
            weight_imaging = subtab.getcol("IMAGING_WEIGHT")
        except:
            try:
                weight_imaging = subtab.getcol("IMAGING_WEIGHT_SPECTRUM")  # (nmeas, nchan, ncorr)
            except:
                pass

        subtab.close()

        if args.average_channels:
            # Continuum mode: one visibility per row, noise-weighted average
            # of the SPW's channels. Var(I_ch) ~ 1/s_ch with s_ch = w1+w4, so
            # the weighted mean has effective weight sum(s_ch).
            if weight_ch is not None:
                w1 = weight_ch[:, :, 0]
                w4 = weight_ch[:, :, ncorr - 1]
            else:
                w1 = np.repeat(weight[:, [0]], nchan_spw, axis=1)
                w4 = np.repeat(weight[:, [ncorr - 1]], nchan_spw, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                y_cube = (w1 * data[:, :, 0] + w4 * data[:, :, ncorr - 1]) / (w1 + w4)
            s_cube = w1 + w4
            flag_pol = flag_cube[:, :, 0] + flag_cube[:, :, ncorr - 1]
            valid = ((flag_pol == 0) & (np.abs(y_cube) > 0)
                     & np.isfinite(y_cube) & (s_cube > 0))
            nvalid = valid.sum(axis=1)
            s_sum = (valid * s_cube).sum(axis=1)
            y_num = (valid * s_cube * np.nan_to_num(y_cube)).sum(axis=1)
            keep = (nvalid >= 1) & (s_sum > 0) & (flag_row == 0)
            nmeasflag = int(np.sum(keep))
            if nmeasflag > 0:
                y_avg = y_num[keep] / s_sum[keep]
                nu_mean = ((valid * freqs_spw[None, :]).sum(axis=1)[keep]
                           / nvalid[keep])
                lam_avg = c / nu_mean
                all_y.append(np.asarray(y_avg).reshape(-1, 1))
                all_u.append(np.reshape(uvw[keep, 0] / lam_avg, (nmeasflag, 1)))
                all_v.append(np.reshape(uvw[keep, 1] / lam_avg, (nmeasflag, 1)))
                all_w.append(np.reshape(uvw[keep, 2] / lam_avg, (nmeasflag, 1)))
                all_nW.append(np.reshape(np.sqrt(s_sum[keep]), (nmeasflag, 1)))
                if weight_imaging is not None:
                    try:
                        wi = (weight_imaging[:, :, 0] if weight_imaging.ndim == 3
                              else np.repeat(weight_imaging[:, [0]], nchan_spw, axis=1))
                        wi_mean = ((valid * wi).sum(axis=1)[keep] / nvalid[keep])
                        all_nWimag.append(np.sqrt(wi_mean))
                    except Exception:
                        pass
            print(f"INFO: SPW {spw_id} done ({nchan_spw} ch -> 1 avg ch, "
                  f"{nmeas} rows -> {nmeasflag} vis)")
            continue

        channels = range(nchan_spw) if args.all_channels else [freqid]
        for ch in channels:
            freq_ch = freqs_spw[ch]
            d_ch = data[:, ch, :]
            if weight_ch is not None:
                w1 = weight_ch[:, ch, 0]
                w4 = weight_ch[:, ch, ncorr - 1]
            else:
                w1 = weight[:, 0]
                w4 = weight[:, ncorr - 1]
            y_ch = (w1 * d_ch[:, 0] + w4 * d_ch[:, ncorr - 1]) / (w1 + w4)
            weight_natural = w1 + w4

            flag = np.reshape(flag_cube[:, ch, 0] + flag_cube[:, ch, ncorr - 1], (nmeas,))
            flag_data = (np.absolute(y_ch) == 0).astype(float)
            keep = np.reshape(np.array((flag + flag_data + flag_row) == False), (nmeas,))
            nmeasflag = int(np.sum(keep))
            if nmeasflag == 0:
                continue

            lam = c / freq_ch
            all_y.append(np.asarray(y_ch[keep]).reshape(-1, 1))
            all_u.append(np.reshape(uvw[keep, 0] / lam, (nmeasflag, 1)))
            all_v.append(np.reshape(uvw[keep, 1] / lam, (nmeasflag, 1)))
            all_w.append(np.reshape(uvw[keep, 2] / lam, (nmeasflag, 1)))
            all_nW.append(np.reshape(np.sqrt(weight_natural[keep]), (nmeasflag, 1)))

            if weight_imaging is not None:
                try:
                    wi = weight_imaging[:, ch, 0] if weight_imaging.ndim == 3 else weight_imaging[:, 0]
                    all_nWimag.append(np.sqrt(wi[keep]))
                except Exception:
                    pass

        print(f"INFO: SPW {spw_id} done ({nchan_spw} ch, {nmeas} rows -> {nmeasflag} vis/ch)")

    y = np.concatenate(all_y)
    u = np.concatenate(all_u)
    v = np.concatenate(all_v)
    w = np.concatenate(all_w)
    nW = np.concatenate(all_nW)
    nWimag = np.concatenate(all_nWimag) if len(all_nWimag) > 0 else []

    ## max projected baseline (for pixel size)
    uv_dist = np.sqrt(u ** 2 + v ** 2)
    
    if args.uv_cutoff is not None:
        keep = uv_dist.flatten() <= args.uv_cutoff
        print(f"INFO: UV cutoff applied. Keeping {np.sum(keep)}/{len(keep)} measurements.")
        y = y[keep]
        u = u[keep]
        v = v[keep]
        w = w[keep]
        nW = nW[keep]
        if len(nWimag) > 0:
            nWimag = nWimag[keep]
        uv_dist = uv_dist[keep]

    maxProjBaseline = np.max(uv_dist).astype(float)
    nominal_pixelsize = (180.0 * 3600.0 / np.pi) * (1.0 / (2.0 * maxProjBaseline))

    data_dir = os.path.dirname(os.path.abspath(msfile))
    suffix = "_allch" if args.all_channels else ("_avg" if args.average_channels else "")
    datamatfile = os.path.join(data_dir, f"{srcname}{suffix}.mat")
    print("INFO: Saving data ..Freq %s" % freqid)


    sio.savemat(
        datamatfile,
        {
            "frequency": frequency,
            "y": y,  # data (Stokes I)
            "u": u,  # u coordinate (in units of the wavelength)
            "v": v,  # v coordinate (in units of the wavelength)
            "w": w,  # w coordinate  (in units of the wavelength)
            "nW": nW,  # 1/sigma: square root of natural weights
            "nWimag": nWimag,  # square root of the imaging weights if available (Briggs or uniform)
            "maxProjBaseline": maxProjBaseline,  # max projected baseline  (in units of the wavelength)
            "nominal_pixelsize": nominal_pixelsize, # Default pixel size (super-resolution factor = 1)
        },
    )
    print("INFO: Data .mat file saved:  %s" % datamatfile)
    print("DONE.")


if __name__ == "__main__":
    main()
