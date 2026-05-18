"""
Re-referencing methods for protein backbone chemical shifts.

reref_panav_  — probabilistic re-referencing (Wishart lab 2005, 2010)
reref_lacs_   — LACS re-referencing with BMRB outlier filtering (Wang & Markley 2009, v1)
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import HuberRegressor

from .utils.reref_utils import apply_offset, get_ss_probs, get_offset_panav
from .utils.chemshift_utils import get_secondary_shift, get_other_csi


# ── Constants ────────────────────────────────────────────────────────────────

_EXCLUDE = {'GLY', 'CYS', 'PRO'}

_THRESH_C  = 0.10
_THRESH_N  = 0.70
_THRESH_HN = 0.12

_SLOPE_TOL = {
    'N': 0.100,
    'H': 0.020,
}
_MIN_N         = 66
_MIN_FRAC_WIDE = 0.15

_ATOM_PARAMS = {
    'N': (-0.400, 0.050, _THRESH_N),
    'H': (-0.070, 0.010, _THRESH_HN),
}


# ── C' random coil values (Wishart et al. 1995) ──────────────────────────────

_RC_C_PRIME = {
    'ALA': 177.8, 'ARG': 176.3, 'ASN': 175.2, 'ASP': 176.3,
    'CYS': 174.6, 'GLN': 176.0, 'GLU': 176.6, 'GLY': 174.9,
    'HIS': 174.1, 'ILE': 176.4, 'LEU': 177.6, 'LYS': 176.6,
    'MET': 176.3, 'PHE': 175.8, 'PRO': 177.3, 'SER': 174.6,
    'THR': 174.7, 'TRP': 176.1, 'TYR': 175.9, 'VAL': 176.3,
}

_AA_1TO3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP',
    'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY',
    'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
    'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER',
    'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}

# ── BMRB full-database statistics ────────────────────────────────────────────
# Source: https://bmrb.io/ref_info/csstats.php?set=full&restype=aa
# Format: (mean, std) — GLY has no CB entry.
# Fetch date: 2026-04-17
_BMRB_STATS = {
    'ALA': {'N': (123.395,  5.602), 'H': (8.197, 0.878), 'CA': (53.132, 2.603), 'CB': (19.023, 2.828), 'C': (177.742, 3.443)},
    'ARG': {'N': (120.871,  4.520), 'H': (8.240, 0.971), 'CA': (56.756, 3.136), 'CB': (30.664, 2.374), 'C': (176.423, 3.129)},
    'ASN': {'N': (118.945,  5.194), 'H': (8.353, 1.864), 'CA': (53.517, 3.230), 'CB': (38.704, 3.282), 'C': (175.208, 3.308)},
    'ASP': {'N': (120.727,  4.449), 'H': (8.295, 0.579), 'CA': (54.664, 2.578), 'CB': (40.878, 2.391), 'C': (176.354, 3.351)},
    'CYS': {'N': (120.340, 16.415), 'H': (8.372, 0.693), 'CA': (57.969, 3.489), 'CB': (33.451, 6.662), 'C': (174.723, 3.256)},
    'GLN': {'N': (120.039,  4.327), 'H': (8.243, 1.806), 'CA': (56.518, 2.507), 'CB': (29.171, 2.372), 'C': (176.306, 8.487)},
    'GLU': {'N': (120.763,  4.432), 'H': (8.331, 0.786), 'CA': (57.303, 3.061), 'CB': (29.987, 2.908), 'C': (176.830, 3.952)},
    'GLY': {'N': (109.636,  6.387), 'H': (8.327, 0.851), 'CA': (45.360, 2.055),                        'C': (173.831, 3.460)},
    'HIS': {'N': (119.695,  5.022), 'H': (8.258, 1.202), 'CA': (56.482, 3.212), 'CB': (30.318, 2.994), 'C': (175.128, 4.436)},
    'ILE': {'N': (121.431,  5.824), 'H': (8.256, 0.686), 'CA': (61.627, 3.231), 'CB': (38.553, 2.764), 'C': (175.832, 4.220)},
    'LEU': {'N': (121.962,  7.212), 'H': (8.218, 0.718), 'CA': (55.643, 2.202), 'CB': (42.209, 1.987), 'C': (177.010, 3.415)},
    'LYS': {'N': (121.069,  4.699), 'H': (8.190, 1.483), 'CA': (56.935, 3.015), 'CB': (32.762, 2.728), 'C': (176.621, 5.198)},
    'MET': {'N': (120.152,  4.893), 'H': (8.258, 1.147), 'CA': (56.112, 2.265), 'CB': (32.954, 3.016), 'C': (176.192, 3.140)},
    'PHE': {'N': (120.401,  5.403), 'H': (8.332, 0.727), 'CA': (58.093, 3.591), 'CB': (39.901, 3.346), 'C': (175.441, 2.863)},
    'PRO': {                                              'CA': (63.319, 3.304), 'CB': (31.871, 2.868), 'C': (176.669, 3.918)},
    'SER': {'N': (116.321,  4.059), 'H': (8.277, 0.685), 'CA': (58.664, 3.336), 'CB': (63.721, 4.455), 'C': (174.547, 3.511)},
    'THR': {'N': (115.383,  6.022), 'H': (8.226, 0.630), 'CA': (62.193, 2.705), 'CB': (69.607, 5.071), 'C': (174.453, 3.796)},
    'TRP': {'N': (121.634,  5.828), 'H': (8.264, 0.783), 'CA': (57.727, 4.422), 'CB': (30.030, 4.376), 'C': (176.040, 5.442)},
    'TYR': {'N': (120.648, 10.623), 'H': (8.282, 0.887), 'CA': (58.126, 2.967), 'CB': (39.267, 2.948), 'C': (175.406, 4.256)},
    'VAL': {'N': (121.154,  6.848), 'H': (8.267, 0.672), 'CA': (62.483, 3.122), 'CB': (32.689, 2.189), 'C': (175.639, 3.203)},
}

_N_STD_OUTLIER = 4


# ── Local helpers ─────────────────────────────────────────────────────────────

def _is_outlier(comp_id, atom_id, val, n_std=_N_STD_OUTLIER):
    if pd.isna(val):
        return False
    res_stats = _BMRB_STATS.get(comp_id.upper(), {})
    atom_stats = res_stats.get(atom_id.upper())
    if atom_stats is None:
        return True
    mean, std = atom_stats
    return abs(val - mean) > n_std * std


def _tag_outliers(df, n_std=_N_STD_OUTLIER):
    outlier_mask = df.apply(
        lambda row: _is_outlier(row['Comp_ID'], row['Atom_ID'], row['Val'], n_std),
        axis=1,
    )
    df.loc[outlier_mask, 'reref_mask'] = False


def _get_csi(row, df):
    data = df.loc[df.Seq_ID == row['Seq_ID']]
    ca_rows = data[data['Atom_ID'] == 'CA']
    cb_rows = data[data['Atom_ID'] == 'CB']
    ca_sec = get_secondary_shift(ca_rows.iloc[0]) if len(ca_rows) else np.nan
    cb_sec = get_secondary_shift(cb_rows.iloc[0]) if len(cb_rows) else np.nan
    if np.isfinite(ca_sec) and np.isfinite(cb_sec):
        return ca_sec - cb_sec
    return np.nan


def _huber_fit(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return None, None
    model = HuberRegressor(fit_intercept=True, max_iter=300)
    model.fit(x.reshape(-1, 1), y)
    return float(model.coef_[0]), float(model.intercept_)


def _bounded_slope_fit(x, y, slope_lo, slope_hi):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return (slope_lo + slope_hi) / 2.0, 0.0
    xc, yc = x - x.mean(), y - y.mean()
    res = minimize_scalar(
        lambda s: float(np.sum((yc - s * xc) ** 2)),
        bounds=(slope_lo, slope_hi),
        method='bounded',
    )
    s_opt = float(res.x)
    return s_opt, float(y.mean() - s_opt * x.mean())


def _piecewise_offset(x, y):
    """Two-segment robust fit per Wang et al. 2005 Equation 1."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return np.nan
    intercepts = []
    for seg_mask in [x >= 0, x < 0]:
        xs, ys = x[seg_mask], y[seg_mask]
        _, intercept = _huber_fit(xs, ys)
        if intercept is not None:
            intercepts.append(intercept)
    if not intercepts:
        return np.nan
    offset = float(np.mean(intercepts))
    return 0.0 if abs(offset) <= _THRESH_C else offset


def _lacs_offset_linear(x, y, expected_slope, slope_tight, threshold, slope_tol):
    """Wang & Markley 2009 four-step procedure for N and H."""
    mask = ~(np.isnan(x) | np.isnan(y))
    xc, yc = x[mask], y[mask]
    n = len(xc)
    if n < 20:
        return np.nan
    slope, intercept = _huber_fit(xc, yc)
    if slope is None:
        return np.nan
    if abs(slope - expected_slope) <= slope_tol:
        offset = intercept
    else:
        n_wide    = int(np.sum(np.abs(xc) > 2))
        frac_wide = n_wide / n
        if n < _MIN_N or frac_wide < _MIN_FRAC_WIDE:
            _, offset = _bounded_slope_fit(
                xc, yc,
                slope_lo=expected_slope - slope_tight,
                slope_hi=expected_slope + slope_tight,
            )
        else:
            offset = intercept
    return 0.0 if abs(offset) <= threshold else offset


def _c_prime_secondary_shift(row):
    rc = _RC_C_PRIME.get(row['Comp_ID'].upper(), np.nan)
    val = row['Val']
    if np.isnan(rc) or np.isnan(val):
        return np.nan
    return val - rc


def _single_line_offset(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    xc, yc = x[mask], y[mask]
    if len(xc) < 20:
        return np.nan
    _, intercept = _huber_fit(xc, yc)
    if intercept is None:
        return np.nan
    return 0.0 if abs(intercept) <= _THRESH_C else intercept


def _prepare_fit_data(df_atom, x_col):
    df_fit = df_atom[
        ~df_atom['Comp_ID'].str.upper().isin(_EXCLUDE) &
        (df_atom['reref_mask'] == True)
    ].copy()
    return df_fit[x_col].values, df_fit['secondary_shift'].values


# ── Public API ────────────────────────────────────────────────────────────────

def reref_panav_(df_0, n_iters=2):
    """
    Probabilistic re-referencing (Wishart lab 2005, 2010).
    Original values are copied to `orig` column. Corrected values in `Val`.
    """
    df = df_0.copy()
    rest = df.loc[~df.Atom_ID.isin(['N', 'CA', 'CB'])]
    df = df.loc[df.Atom_ID.isin(['HA', 'HA2', 'H', 'N', 'CA', 'CB'])]
    df["Atom_ID"] = df["Atom_ID"].replace("HA2", "HA")
    df['orig'] = df['Val'].copy()
    net_offsets = {}
    for j in range(n_iters):
        df[['ss_probs', 'ss_max']] = df.apply(
            lambda row: get_ss_probs(row), axis=1, result_type='expand'
        )
        ha = df.loc[df.Atom_ID == 'HA']
        df['offset'] = df.apply(
            lambda row: get_offset_panav(row, ha.loc[ha.Seq_ID == row['Seq_ID']]), axis=1
        )
        offsets = {}
        for atom in ['N', 'CA', 'CB']:
            vals = np.array(df.loc[df.Atom_ID == atom]['offset'])
            s = np.nanstd(vals)
            m = np.nanmedian(vals)
            vals[np.where(vals > m + 3 * s)] = np.nan
            vals[np.where(vals < m - 3 * s)] = np.nan
            offsets[atom] = np.nanmedian(vals)
            if j == 0:
                net_offsets[atom] = np.nanmedian(vals)
            else:
                net_offsets[atom] += np.nanmedian(vals)
        df['Val'] = df.apply(lambda row: apply_offset(row, offsets), axis=1)
    df = pd.concat([df, rest])
    if df.attrs is None:
        df.attrs = {'PANAV offsets': net_offsets}
    else:
        df.attrs.update({'PANAV offsets': net_offsets})
    return df


def reref_lacs_(df_0, n_std=_N_STD_OUTLIER):
    """
    LACS re-referencing for backbone CA, CB, C', N, and H (v1).

    Before fitting, each observed shift is checked against BMRB full-database
    statistics (mean ± n_std * std). Outlier rows receive reref_mask=False and
    are excluded from the regression.

    reref_mask semantics
    --------------------
    True  — shift is within expected range AND offset was successfully applied.
    False — shift is a statistical outlier OR atom offset could not be computed.
            Val is unchanged; exclude from downstream training.

    Parameters
    ----------
    df_0   : DataFrame with columns Seq_ID, Comp_ID, Atom_ID, Val
    n_std  : outlier threshold in units of BMRB std (default 4)
    """
    df = df_0.copy()
    df = df.loc[:, ~df.columns.duplicated()]

    df['orig'] = df['Val'].copy()

    df['reref_mask'] = True
    _tag_outliers(df, n_std=n_std)

    df['secondary_shift'] = df.apply(lambda row: get_secondary_shift(row), axis=1)
    c_mask = df['Atom_ID'] == 'C'
    df.loc[c_mask, 'secondary_shift'] = df.loc[c_mask].apply(
        _c_prime_secondary_shift, axis=1
    )

    df['csi']      = df.apply(lambda row: _get_csi(row, df), axis=1)
    df['csi_prev'] = df.apply(lambda row: get_other_csi(row, df, -1), axis=1)

    offsets     = {}
    net_offsets = {}

    for atom in ('CA', 'CB'):
        x, y = _prepare_fit_data(df[df['Atom_ID'] == atom], 'csi')
        offset = _piecewise_offset(x, y)
        offsets[atom]     = offset
        net_offsets[atom] = offset

    x, y = _prepare_fit_data(df[df['Atom_ID'] == 'C'], 'csi')
    offset = _single_line_offset(x, y)
    offsets['C']     = offset
    net_offsets['C'] = offset

    for atom in ('N', 'H'):
        expected_slope, slope_tight, threshold = _ATOM_PARAMS[atom]
        x, y = _prepare_fit_data(df[df['Atom_ID'] == atom], 'csi_prev')
        offset = _lacs_offset_linear(
            x, y, expected_slope, slope_tight, threshold, _SLOPE_TOL[atom]
        )
        offsets[atom]     = offset
        net_offsets[atom] = offset

    valid_offsets = {
        atom: v for atom, v in offsets.items()
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    }

    skip = not bool(valid_offsets)

    for atom, v in offsets.items():
        if atom not in valid_offsets:
            df.loc[df['Atom_ID'] == atom, 'reref_mask'] = False

    df['Val'] = df.apply(
        lambda row: apply_offset(row, valid_offsets) if row['reref_mask'] else row['Val'],
        axis=1,
    )

    net_offsets['method'] = 'LACS_v1'

    return df, skip, valid_offsets
