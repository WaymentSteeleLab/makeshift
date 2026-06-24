"""
Re-referencing methods for protein backbone chemical shifts.

Public entry point:
    reref(df, method)  —  'lacs' or 'panav'
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import HuberRegressor

from .utils.reref_utils import apply_offset
from .utils.chemshift_utils import get_secondary_shift, get_csi
from .utils.tables import get_panav_distns, get_bmrb_stats, get_c_prime_rc

_PANAV_REF  = get_panav_distns()
_BMRB_STATS = get_bmrb_stats()
_RC_C_PRIME = get_c_prime_rc()

# ── Algorithm constants ───────────────────────────────────────────────────────

_EXCLUDE = {'GLY', 'CYS', 'PRO'}

_THRESH_C  = 0.10
_THRESH_N  = 0.70
_THRESH_HN = 0.12

_SLOPE_TOL = {'N': 0.100, 'H': 0.020}
_MIN_N         = 66
_MIN_FRAC_WIDE = 0.15

_ATOM_PARAMS = {
    'N': (-0.400, 0.050, _THRESH_N),
    'H': (-0.070, 0.010, _THRESH_HN),
}

_N_STD_OUTLIER = 4
_NO_REF = {('PRO', 'N'), ('PRO', 'H'), ('GLY', 'CB')}

# ── Local helpers ─────────────────────────────────────────────────────────────
def _has_ref(comp_id, atom_id):
    if (comp_id, atom_id) in _NO_REF:
        return False
    try:
        for ss in ['C', 'H', 'E']:
            mean, std = _PANAV_REF[comp_id][ss][atom_id]
            if pd.isna(mean) or pd.isna(std):
                return False
        return True
    except KeyError:
        return False

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


# ── PANAV local helpers ───────────────────────────────────────────────────────

def _gaussian(x, mu, sigma):
    if sigma == 0:
        return 0.0
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def _panav_ss_probs(row):
    if not _has_ref(row['Comp_ID'], row['Atom_ID']):
        return np.nan, np.nan
    scores = []
    for ss in ['C', 'H', 'E']:
        mean, std = _PANAV_REF[row['Comp_ID']][ss][row['Atom_ID']]
        scores.append(_gaussian(row['Val'], mean, std))
    if scores == [0.0, 0.0, 0.0]:
        return np.nan, np.nan
    probs = np.array(scores) / np.sum(scores)
    selected = ['C', 'H', 'E'][np.argmax(scores)]
    if row['Atom_ID'] == 'HA':
        mean, std = _PANAV_REF[row['Comp_ID']][selected][row['Atom_ID']]
        if abs(row['Val'] - mean) > 4 * std:
            return np.nan, np.nan
    return probs, selected


def _panav_get_offset(row, corresponding_ha):
    if not _has_ref(row['Comp_ID'], row['Atom_ID']):
        return np.nan, np.nan
    if len(corresponding_ha) == 0:
        return np.nan, np.nan
    ss = corresponding_ha.iloc[0]['ss_max']
    if pd.isna(ss):
        return np.nan, np.nan
    mean, _ = _PANAV_REF[row['Comp_ID']][ss][row['Atom_ID']]
    return row['Val'] - mean, ss


def _panav_judge_outlier(row):
    if not _has_ref(row['Comp_ID'], row['Atom_ID']):
        return True
    if pd.isna(row['offset']):
        return all(
            abs(row['Val'] - _PANAV_REF[row['Comp_ID']][ss][row['Atom_ID']][0])
            > 4 * _PANAV_REF[row['Comp_ID']][ss][row['Atom_ID']][1]
            for ss in ['C', 'H', 'E']
        )
    mean, std = _PANAV_REF[row['Comp_ID']][row['ss_max']][row['Atom_ID']]
    return abs(row['offset']) > 4 * std


# ── Public API ────────────────────────────────────────────────────────────────

_LACS_ATOMS  = ('CA', 'CB', 'C', 'N', 'H')
_PANAV_ATOMS = ('N', 'CA', 'CB', 'C')


def _reref_panav(df_0):
    df = df_0.copy()
    df["Atom_ID"] = df["Atom_ID"].replace("HA2", "HA")
    df['orig'] = df['Val'].copy()
    df['outlier_1'] = False
    df['outlier_2'] = False

    check = {atom: True for atom in _PANAV_ATOMS}
    cumulative_offsets = {atom: 0.0 for atom in _PANAV_ATOMS}

    for round_i in range(2):
        df[['ss_probs', 'ss_max']] = df.apply(
            lambda row: _panav_ss_probs(row), axis=1, result_type='expand'
        )
        ha = df.loc[df.Atom_ID == 'HA']
        df[['offset', 'ss_max']] = df.apply(
            lambda row: _panav_get_offset(row, ha.loc[ha.Seq_ID == row['Seq_ID']]),
            axis=1, result_type='expand',
        )

        outlier_col = f'outlier_{round_i + 1}'
        df[outlier_col] = df.apply(lambda row: _panav_judge_outlier(row), axis=1)

        offsets = {}
        for atom in _PANAV_ATOMS:
            mask = (df.Atom_ID == atom) & (~df['outlier_1'])
            if round_i == 1:
                mask &= ~df['outlier_2']
            vals = np.array(df.loc[mask, 'offset'])

            if np.isnan(vals).all():
                offsets[atom] = None
                check[atom] = False
                continue

            s = np.nanstd(vals)
            m = np.nanmean(vals)
            vals[vals > m + 3 * s] = np.nan
            vals[vals < m - 3 * s] = np.nan

            if (~np.isnan(vals)).sum() < 25:
                offsets[atom] = None
                check[atom] = False
                continue

            offsets[atom] = float(np.nanmean(vals))

        offsets_apply = {k: (0 if v is None else v) for k, v in offsets.items()}
        df['Val'] = df.apply(lambda row: apply_offset(row, offsets_apply), axis=1)

        for atom in _PANAV_ATOMS:
            if offsets.get(atom) is not None:
                cumulative_offsets[atom] += offsets[atom]

        failed = [atom for atom, ok in check.items() if not ok]
        df.loc[df['Atom_ID'].isin(failed), outlier_col] = True

    total_offsets = {atom: (cumulative_offsets[atom] if check[atom] else None)
                    for atom in _PANAV_ATOMS}
    return df, check, total_offsets


def _reref_lacs(df_0, n_std=_N_STD_OUTLIER):
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

    df['csi']      = df.apply(lambda row: get_csi(row, df, strict=True), axis=1)
    df['csi_prev'] = df.apply(lambda row: get_csi(row, df, offset=-1), axis=1)

    raw_offsets = {}

    for atom in ('CA', 'CB'):
        x, y = _prepare_fit_data(df[df['Atom_ID'] == atom], 'csi')
        raw_offsets[atom] = _piecewise_offset(x, y)

    x, y = _prepare_fit_data(df[df['Atom_ID'] == 'C'], 'csi')
    raw_offsets['C'] = _single_line_offset(x, y)

    for atom in ('N', 'H'):
        expected_slope, slope_tight, threshold = _ATOM_PARAMS[atom]
        x, y = _prepare_fit_data(df[df['Atom_ID'] == atom], 'csi_prev')
        raw_offsets[atom] = _lacs_offset_linear(
            x, y, expected_slope, slope_tight, threshold, _SLOPE_TOL[atom]
        )

    def _fitted(v):
        return v is not None and not (isinstance(v, float) and np.isnan(v))

    valid_offsets = {atom: v for atom, v in raw_offsets.items() if _fitted(v)}
    check   = {atom: _fitted(raw_offsets.get(atom)) for atom in _LACS_ATOMS}
    offsets = {atom: raw_offsets.get(atom) if _fitted(raw_offsets.get(atom)) else None
               for atom in _LACS_ATOMS}

    for atom in _LACS_ATOMS:
        if not check[atom]:
            df.loc[df['Atom_ID'] == atom, 'reref_mask'] = False

    df['Val'] = df.apply(
        lambda row: apply_offset(row, valid_offsets) if row['reref_mask'] else row['Val'],
        axis=1,
    )

    return df, check, offsets


def reref(df, method, n_std=_N_STD_OUTLIER):
    """
    Re-reference backbone chemical shifts using LACS or PANAV.

    Filters the input to relevant atoms, averages GLY HA2/HA pairs, then
    runs the requested re-referencing algorithm.

    Parameters
    ----------
    df     : DataFrame returned by get_chem_shifts()
    method : 'lacs' or 'panav'
    n_std  : (LACS only) outlier threshold in units of BMRB std (default 4)

    Returns
    -------
    df      : DataFrame with corrected Val column. Also adds:
                - 'orig'       : original Val before correction
                - 'reref_mask' : (LACS) True if shift is an inlier and was corrected;
                                 False if it was a statistical outlier or fitting failed
                - 'outlier_1',
                  'outlier_2'  : (PANAV) per-round outlier flags
    check   : dict {atom: bool} — True if re-referencing succeeded for that atom.
              LACS atoms: CA, CB, C, N, H
              PANAV atoms: N, CA, CB, C
    offsets : dict {atom: float | None} — offset applied to each atom in the final
              fitting round; None if referencing failed for that atom.
              For LACS this is the single fitted offset.
              For PANAV this is the round-2 offset (the residual correction after
              round 1 has already been applied).
    """
    if method not in ('lacs', 'panav'):
        raise ValueError(f"method must be 'lacs' or 'panav', got {method!r}")

    df_i = df.copy()
    df_i = df_i.loc[
        df_i.Atom_ID.isin(['H', 'HA', 'N', 'CA', 'CB', 'C']) |
        df_i.Atom_ID.str.contains('^HA', na=False)
    ]

    # Average GLY HA/HA2 pairs into a single HA per residue
    gly_mask = (df_i['Comp_ID'] == 'GLY') & df_i['Atom_ID'].str.contains('HA')
    for seq_id, group in df_i[gly_mask].groupby('Seq_ID'):
        mean_val = group['Val'].mean()
        mask = (df_i['Seq_ID'] == seq_id) & gly_mask
        df_i.loc[mask, 'Val'] = mean_val
        indices = df_i.loc[mask].index
        df_i.loc[indices[0], 'Atom_ID'] = 'HA'
        if len(indices) > 1:
            df_i.loc[indices[1:], 'Atom_ID'] = 'HA2'

    df_i = df_i.loc[df_i.Atom_ID.isin(['H', 'HA', 'N', 'CA', 'CB', 'C'])].copy()

    if df_i.empty:
        return None, None, None

    if method == 'lacs':
        return _reref_lacs(df_i, n_std=n_std)
    else:
        return _reref_panav(df_i)
