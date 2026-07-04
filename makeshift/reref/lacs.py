"""
LACS re-referencing (Wang & Markley 2009; Wang et al. 2005).

Per-atom offset from the intercept of secondary shift vs. CSI (CA, CB, C) or
vs. the previous residue's CSI (N, H), via robust (Huber) and bounded-slope
fits. 
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import HuberRegressor

from ..data.tables import get_bmrb_stats, get_c_prime_rc

_BMRB_STATS = get_bmrb_stats()     # {residue: {atom: (mean, std)}}
_RC_C_PRIME = get_c_prime_rc()     # {residue: float}

_EXCLUDE = {"GLY", "CYS", "PRO"}

_THRESH_C = 0.10
_THRESH_N = 0.70
_THRESH_HN = 0.12

_SLOPE_TOL = {"N": 0.100, "H": 0.020}
_MIN_N = 66
_MIN_FRAC_WIDE = 0.15

# atom -> (expected_slope, slope_tight, threshold) for the N/H linear fit
_ATOM_PARAMS = {
    "N": (-0.400, 0.050, _THRESH_N),
    "H": (-0.070, 0.010, _THRESH_HN),
}

_N_STD_OUTLIER = 4
_LACS_ATOMS = ("CA", "CB", "C", "N", "H")


def _fitted(v):
    """True if an offset was successfully produced (not None / not NaN)."""
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def _c_prime_secondary_shift(comp_id, val):
    """Secondary shift for the carbonyl C', which has its own random-coil table."""
    rc = _RC_C_PRIME.get(comp_id.upper(), np.nan)
    if np.isnan(rc) or np.isnan(val):
        return np.nan
    return val - rc


def _is_outlier(comp_id, atom_id, val, n_std):
    """Flag shifts far from the BMRB full-database mean for that atom."""
    if pd.isna(val):
        return False
    atom_stats = _BMRB_STATS.get(comp_id.upper(), {}).get(atom_id.upper())
    if atom_stats is None:
        return True
    mean, std = atom_stats
    return abs(val - mean) > n_std * std


def _huber_fit(x, y):
    """Robust linear fit. Returns (slope, intercept), or (None, None) if < 3 pts."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return None, None
    model = HuberRegressor(fit_intercept=True, max_iter=300)
    model.fit(x.reshape(-1, 1), y)
    return float(model.coef_[0]), float(model.intercept_)


def _bounded_slope_fit(x, y, slope_lo, slope_hi):
    """Least-squares intercept with the slope constrained to [slope_lo, slope_hi]."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return (slope_lo + slope_hi) / 2.0, 0.0
    xc, yc = x - x.mean(), y - y.mean()
    res = minimize_scalar(
        lambda s: float(np.sum((yc - s * xc) ** 2)),
        bounds=(slope_lo, slope_hi),
        method="bounded",
    )
    s_opt = float(res.x)
    return s_opt, float(y.mean() - s_opt * x.mean())


def _piecewise_offset(x, y):
    """Two-segment robust fit (x >= 0 and x < 0); offset = mean of intercepts."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return np.nan
    intercepts = []
    for seg_mask in (x >= 0, x < 0):
        _, intercept = _huber_fit(x[seg_mask], y[seg_mask])
        if intercept is not None:
            intercepts.append(intercept)
    if not intercepts:
        return np.nan
    offset = float(np.mean(intercepts))
    return 0.0 if abs(offset) <= _THRESH_C else offset


def _single_line_offset(x, y):
    """Single robust-fit intercept (used for C'), zeroed below threshold."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 20:
        return np.nan
    _, intercept = _huber_fit(x, y)
    if intercept is None:
        return np.nan
    return 0.0 if abs(intercept) <= _THRESH_C else intercept


def _lacs_offset_linear(x, y, expected_slope, slope_tight, threshold, slope_tol):
    """Four-step N/H procedure: trust the robust intercept when the slope is
    near expected, otherwise fall back to a bounded-slope fit on sparse data."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 20:
        return np.nan
    slope, intercept = _huber_fit(x, y)
    if slope is None:
        return np.nan
    if abs(slope - expected_slope) <= slope_tol:
        offset = intercept
    else:
        n_wide = int(np.sum(np.abs(x) > 2))
        frac_wide = n_wide / n
        if n < _MIN_N or frac_wide < _MIN_FRAC_WIDE:
            _, offset = _bounded_slope_fit(
                x, y,
                slope_lo=expected_slope - slope_tight,
                slope_hi=expected_slope + slope_tight,
            )
        else:
            offset = intercept
    return 0.0 if abs(offset) <= threshold else offset


def _prepare_fit_data(df_atom, x_col):
    """x (CSI) and y (secondary shift) for one atom, dropping excluded residues
    and statistical outliers."""
    keep = (
        ~df_atom["Comp_ID"].str.upper().isin(_EXCLUDE)
        & (df_atom["reref_mask"] == True)  # noqa: E712
    )
    df_fit = df_atom[keep]
    return df_fit[x_col].to_numpy(), df_fit["secondary_shift"].to_numpy()


# Entry point 

def reref_lacs(df, n_std=_N_STD_OUTLIER):
    """Compute per-atom LACS offsets for a backbone shift table."""
    
    from ..chemshift import ChemicalShifts  # reuse secondary-shift / CSI logic

    df = df.loc[:, ~df.columns.duplicated()].copy()
    df["reref_mask"] = True
    outliers = df.apply(
        lambda r: _is_outlier(r["Comp_ID"], r["Atom_ID"], r["Val"], n_std), axis=1
    )
    df.loc[outliers, "reref_mask"] = False

    # secondary shift: CA/CB/N/H from the random-coil table; C' from its own
    df["secondary_shift"] = df.apply(ChemicalShifts._secondary_shift, axis=1)
    c_mask = df["Atom_ID"] == "C"
    df.loc[c_mask, "secondary_shift"] = df.loc[c_mask].apply(
        lambda r: _c_prime_secondary_shift(r["Comp_ID"], r["Val"]), axis=1
    )

    # CSI (CA - CB secondary shift) per residue, strict so a CA-only value
    # never enters the fit; csi_prev is the previous residue's CSI
    work = ChemicalShifts(df)
    df["csi"] = df.apply(lambda r: work._csi_raw(r, strict=True), axis=1)
    csi_by_seq = df.drop_duplicates("Seq_ID").set_index("Seq_ID")["csi"]
    df["csi_prev"] = df["Seq_ID"].map(lambda s: csi_by_seq.get(s - 1, np.nan))

    raw = {}
    for atom in ("CA", "CB"):
        x, y = _prepare_fit_data(df[df["Atom_ID"] == atom], "csi")
        raw[atom] = _piecewise_offset(x, y)

    x, y = _prepare_fit_data(df[df["Atom_ID"] == "C"], "csi")
    raw["C"] = _single_line_offset(x, y)

    for atom in ("N", "H"):
        expected_slope, slope_tight, threshold = _ATOM_PARAMS[atom]
        x, y = _prepare_fit_data(df[df["Atom_ID"] == atom], "csi_prev")
        raw[atom] = _lacs_offset_linear(
            x, y, expected_slope, slope_tight, threshold, _SLOPE_TOL[atom]
        )

    check = {atom: _fitted(raw.get(atom)) for atom in _LACS_ATOMS}
    offsets = {atom: (raw[atom] if check[atom] else None) for atom in _LACS_ATOMS}
    return offsets, check