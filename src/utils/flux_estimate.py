"""Total-flux estimator from a Measurement Set (extrapolation to uv=0).

For a positive source V(0) = total flux. No zero baseline exists and short
baselines resolve out extended emission, so the short-baseline visibility
trend is binned radially and a low-order polynomial in r^2 is extrapolated
to r=0 (shortest 1% of rows, 8 bins). Fallback: coherent mean over the
shortest 10% (rejects fits that are non-finite/non-positive or above
ABSURD_FACTOR x max|V|).

Returns (F_est, path) with path in {"fit", "fallback"}.
"""
import numpy as np

FIT_DEGREE = 2          # polynomial in r^2 fitted to short-baseline bins
FIT_FRAC = 0.01         # shortest 1% of baselines used for the fit
FALLBACK_FRAC = 0.10    # shortest 10% used by the fallback coherent mean
N_BINS = 8              # radial bins for the fit
ABSURD_FACTOR = 10.0    # reject fits above this x global max|V|


def _read_ms(ms_path):
    import casacore.tables as ct

    with ct.table(ms_path, readonly=True) as t:
        uvw = t.getcol("UVW")
        data = t.getcol("DATA")
        flag = t.getcol("FLAG")
        flag_row = t.getcol("FLAG_ROW")
    return uvw, data, flag, flag_row


def _coherent_mean(data, flag, sel):
    """Complex mean over selected rows with per-(chan,pol) flags as NaN."""
    d = data[sel]
    f = flag[sel]
    d = np.where(f, np.complex128(np.nan + 1j * np.nan), d.astype(np.complex128))
    return np.nanmean(d), int(np.isfinite(d).sum())


def estimate_total_flux(ms_path, fit_frac=FIT_FRAC, fallback_frac=FALLBACK_FRAC):
    """Estimate total flux density (Jy); returns (F_est, path)."""
    uvw, data, flag, flag_row = _read_ms(ms_path)
    radius = np.hypot(uvw[:, 0], uvw[:, 1])
    valid = ~flag_row

    # Global max|V| over unflagged data (absurdity bound for the fit).
    d_all = np.where(flag, np.nan, data.astype(np.complex128))[valid]
    max_vis = float(np.nanmax(np.abs(d_all))) if np.isfinite(d_all).sum() else np.nan

    # --- primary: polynomial extrapolation to uv=0 over the shortest baselines
    fit_thresh = np.percentile(radius, 100.0 * fit_frac)
    fit_sel = valid & (radius <= fit_thresh)
    F_est = np.nan
    if fit_sel.sum() >= N_BINS + FIT_DEGREE + 1 and np.isfinite(max_vis):
        edges = np.linspace(0.0, fit_thresh, N_BINS + 1)
        rs, vs = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            bin_sel = fit_sel & (radius > lo if lo > 0 else radius >= 0) & (radius <= hi)
            mean, n = _coherent_mean(data, flag, bin_sel)
            if np.isfinite(mean) and n > 0:
                rs.append(0.5 * (lo + hi))
                vs.append(mean)
        if len(vs) >= FIT_DEGREE + 2:
            r2 = np.asarray(rs) ** 2
            v_real = np.real(vs)
            try:
                coeffs = np.polyfit(r2, v_real, FIT_DEGREE)
                intercept = np.polyval(coeffs, 0.0)
                if (np.isfinite(intercept) and intercept > 0.0
                        and intercept <= ABSURD_FACTOR * max_vis):
                    F_est = float(intercept)
            except (np.linalg.LinAlgError, ValueError):
                pass

    if np.isfinite(F_est):
        return F_est, "fit"

    # --- fallback: coherent mean over the shortest decile
    fb_thresh = np.percentile(radius, 100.0 * fallback_frac)
    fb_sel = valid & (radius <= fb_thresh)
    mean, n = _coherent_mean(data, flag, fb_sel)
    if not np.isfinite(mean) or n == 0:
        raise ValueError("no unflagged short-baseline samples")
    return float(abs(mean)), "fallback"
