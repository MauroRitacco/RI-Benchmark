# script to extra data and relevant information in .mat file
# Author: A. Dabbech
import os
import sys
import argparse
import numpy as np
import scipy.io as sio
import math
from casacore import tables

# constants
c = 299792458


def main():
    parser = argparse.ArgumentParser(description='Extract data file from MS')
    parser.add_argument('--msfile', type=str, default=None, help='Path to MS file')
    parser.add_argument('--srcname', type=str, default='', help='Source name')
    parser.add_argument('--srcid', type=int, default=0, help='Source/Field ID')
    parser.add_argument('--freqid', type=int, default=0, help='Frequency ID')
    parser.add_argument('--uv_cutoff', type=float, default=None, help='Max baseline in wavelengths')
    args = parser.parse_args()
    msfile =  args.msfile
    srcname = args.srcname
    srcid = args.srcid
    freqid = args.freqid

    # get data
    tab = tables.table(msfile)
    print("INFO: MS table columns:", *(tab.colnames()))
    print("INFO: Reading data ..Freq %s" % freqid)
    ## freq : table & freq. channels
    spwtab = tables.table("%s/SPECTRAL_WINDOW"%msfile)
    try:
        freqs = spwtab.getcol("CHAN_FREQ")
        frequency = freqs[0, freqid]
    except RuntimeError:
        freqs = spwtab.getcell("CHAN_FREQ", 0)
        frequency = freqs[freqid]
    spwtab.close()

    ## load remaining specs
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
            freqs = spwtab.getcell("CHAN_FREQ", spw_id)
            spw_freq = freqs[freqid]
        except Exception:
            spw_freq = frequency # fallback to global frequency
        spwtab.close()

        ## natural weights: noise vect:  1/variance
        try:
            weight_ch = subtab.getcol("WEIGHT_SPECTRUM")
            ncorr = weight_ch.shape[2]
            weight_ch = weight_ch[:, freqid, :]
            w1 = weight_ch[:, 0]
            w4 = weight_ch[:, ncorr - 1]
        except:
            weight = subtab.getcol("WEIGHT")
            ncorr = weight.shape[1]
            w1 = weight[:, 0]
            w4 = weight[:, ncorr - 1]

        ## data
        try:
            data = subtab.getcol("CORRECTED_DATA")
        except:
            data = subtab.getcol("DATA")
            
        data = data[:, freqid, :]
        data = (w1 * data[:, 0] + w4 * data[:, ncorr - 1]) / (w1 + w4)
        data = np.reshape(np.array(data), (nmeas, 1))
        weight_natural = np.reshape(np.array(w1 + w4), (nmeas, 1))

        ## flag
        flag_row = subtab.getcol("FLAG_ROW")
        flag_row = (np.reshape(np.array(flag_row), (nmeas,))).astype(float)
        flag = subtab.getcol("FLAG")
        flag = (flag[:, freqid, :]).astype(float)
        flag = np.reshape(np.array(flag[:, 0] + flag[:, ncorr - 1]), (nmeas,))
        flag_data = np.reshape((np.absolute(data) == False).astype(float), (nmeas,))
        flag = (flag + flag_data + flag_row) == False
        flag = np.reshape(np.array(flag), (nmeas,))

        ## u,v,w,
        uvw = subtab.getcol("UVW")

        ## briggs/uniform imaging weights
        weight_imaging = []
        try:
            weight_imaging = subtab.getcol("IMAGING_WEIGHT")
            weight_imaging = weight_imaging[:, 0]
        except:
            try:
                weight_imaging = subtab.getcol("IMAGING_WEIGHT_SPECTRUM")
                weight_imaging = weight_imaging[:, freqid, 0]
            except:
                pass
                
        subtab.close()

        ## applying flags
        y_spw = data[flag]
        nmeasflag = len(y_spw)
        u_spw = np.reshape(uvw[flag, 0] / (c / spw_freq), (nmeasflag, 1))
        v_spw = np.reshape(uvw[flag, 1] / (c / spw_freq), (nmeasflag, 1))
        w_spw = np.reshape(uvw[flag, 2] / (c / spw_freq), (nmeasflag, 1))
        nW_spw = (np.sqrt(weight_natural[flag]))
        
        try:
            nWimag_spw = (np.sqrt(weight_imaging[flag]))
            nWimag_spw = np.reshape(np.array(nWimag_spw), (nmeasflag, 1))
        except:
            nWimag_spw = np.array([])
            
        all_y.append(y_spw)
        all_u.append(u_spw)
        all_v.append(v_spw)
        all_w.append(w_spw)
        all_nW.append(nW_spw)
        if len(nWimag_spw) > 0:
            all_nWimag.append(nWimag_spw)

    y = np.concatenate(all_y)
    u = np.concatenate(all_u)
    v = np.concatenate(all_v)
    w = np.concatenate(all_w)
    nW = np.concatenate(all_nW)
    nWimag = np.concatenate(all_nWimag) if len(all_nWimag) > 0 else []

    ## maximum projected baseline (used for pixel size)
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

    ## save data
    data_dir = os.path.dirname(os.path.abspath(msfile))
    datamatfile = os.path.join(data_dir, f"{srcname}.mat")
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
