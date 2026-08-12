"""
LACS re-referencing (Wang et al. 2005; Wang & Markley 2009).

Per-atom offset from the intercept of secondary shift vs. CSI (CA, CB, C) or
vs. the previous residue's CSI (N, H). Offsets use ``offset ≈ d_ave - d_obs``
(``corrected = Val + offset``).

Nearest-neighbor coil corrections match official BMRB LACS
(``bmrb-io/LACS`` ``ord.m`` / ``ordN.m``):
- CA/CB/C': Wishart et al. 1995 Table 5 when residue i+1 is PRO
  (2005 text names CA/CB; shipping software also corrects CO/C').
- N: Wishart et al. 1995 Table 8 i-1 correction (Wang & Markley 2009).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import HuberRegressor

from ..data.tables import (
    get_bmrb_stats,
    get_c_prime_rc,
    get_random_coil,
    get_rc_pre_pro,
    get_rc_n_prev,
)

_BMRB_STATS = get_bmrb_stats()
_RC_C_PRIME = get_c_prime_rc()
_RC = get_random_coil()
_RC_PRE_PRO = get_rc_pre_pro()
_RC_N_PREV = get_rc_n_prev()
_RC_N_PREV_GLY = _RC_N_PREV.get("GLY", 124.0)

_EXCLUDE = {"GLY", "CYS", "PRO"}

_THRESH_C = 0.10
_THRESH_N = 0.70
_THRESH_HN = 0.12

# N: paper/software "k close to 0.4 ± 0.1". H: official LACS (ordN.m) uses
# |k + 0.07| > 0.02 for the step-2 check (not the ±0.01 step-3 bound).
_SLOPE_TOL = {"N": 0.100, "H": 0.020}
_MIN_N = 66
_MIN_FRAC_WIDE = 0.15

_ATOM_PARAMS = {
    "N": (-0.400, 0.050, _THRESH_N),
    "H": (-0.070, 0.010, _THRESH_HN),
}

_N_STD_OUTLIER = 4
_LACS_ATOMS = ("CA", "CB", "C", "N", "H")


def _comp_at(df, seq_id):
    """Comp_ID at Seq_ID, or None."""
    hit = df.loc[df["Seq_ID"] == seq_id, "Comp_ID"]
    return None if hit.empty else str(hit.iloc[0]).upper()


def _huber_fit(x, y):
    """Robust linear fit. Returns (slope, intercept), or (None, None) if < 3 pts."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return None, None
    model = HuberRegressor(fit_intercept=True, max_iter=300)
    model.fit(x.reshape(-1, 1), y)
    return float(model.coef_[0]), float(model.intercept_)


def _prepare_fit_data(df_atom, x_col, exclude_col="Comp_ID"):
    """
    Drop GLY/CYS/PRO on ``exclude_col`` and statistical outliers.

    For CA/CB/C′ fits, ``exclude_col`` is the atom's own residue (Wang 2005).
    For N/H fits the x-axis is residue i−1's CSI, so callers pass
    ``prev_comp_id`` instead.
    """
    keep = (
        ~df_atom[exclude_col].astype(str).str.upper().isin(_EXCLUDE)
        & (df_atom["reref_mask"] == True)  # noqa: E712
    )
    df_fit = df_atom[keep]
    return df_fit[x_col].to_numpy(), df_fit["secondary_shift"].to_numpy()


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


def _lacs_offset_linear(x, y, expected_slope, slope_tight, threshold, slope_tol):
    """
    Four-step N/H procedure (Wang & Markley 2009).

    Trust the robust intercept when the slope is near expected; otherwise
    fall back to a bounded-slope fit when N_total < 66 or the fraction of
    |CSI| > 2 points is small.
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    slope, intercept = _huber_fit(x, y)
    if slope is None:
        return np.nan
    if abs(slope - expected_slope) <= slope_tol:
        offset = intercept
    else:
        n_wide = int(np.sum(np.abs(x) > 2))
        frac_wide = n_wide / n
        if n < _MIN_N or frac_wide < _MIN_FRAC_WIDE:
            if n < 2:
                offset = 0.0
            else:
                xc, yc = x - x.mean(), y - y.mean()
                res = minimize_scalar(
                    lambda s: float(np.sum((yc - s * xc) ** 2)),
                    bounds=(expected_slope - slope_tight,
                            expected_slope + slope_tight),
                    method="bounded",
                )
                s_opt = float(res.x)
                offset = float(y.mean() - s_opt * x.mean())
        else:
            offset = intercept
    return 0.0 if abs(offset) <= threshold else offset


def reref_lacs(df, n_std=_N_STD_OUTLIER):
    """Compute per-atom LACS offsets for a backbone shift table."""
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df["reref_mask"] = True

    def _is_outlier(row):
        if pd.isna(row["Val"]):
            return False
        stats = _BMRB_STATS.get(str(row["Comp_ID"]).upper(), {}).get(
            str(row["Atom_ID"]).upper()
        )
        if stats is None:
            return True
        mean, std = stats
        return abs(row["Val"] - mean) > n_std * std

    df.loc[df.apply(_is_outlier, axis=1), "reref_mask"] = False

    seq_ids = sorted(df["Seq_ID"].dropna().unique())
    df["next_comp_id"] = df["Seq_ID"].map(
        {s: _comp_at(df, s + 1) for s in seq_ids}
    )
    df["prev_comp_id"] = df["Seq_ID"].map(
        {s: _comp_at(df, s - 1) for s in seq_ids}
    )

    def _secondary_shift(row):
        atom = row["Atom_ID"]
        aa = str(row["Comp_ID"]).upper()
        val = row["Val"]
        if atom in ("CA", "CB"):
            if row["next_comp_id"] == "PRO":
                rc = _RC_PRE_PRO.get(aa, {}).get(atom)
                if rc is not None and not np.isnan(val):
                    return val - rc
            try:
                rc = _RC[aa][atom]
            except KeyError:
                return np.nan
            return np.nan if (np.isnan(rc) or np.isnan(val)) else val - rc
        if atom == "C":
            if row["next_comp_id"] == "PRO":
                rc = _RC_PRE_PRO.get(aa, {}).get("C")
                if rc is not None and not np.isnan(val):
                    return val - rc
            rc = _RC_C_PRIME.get(aa, np.nan)
            return np.nan if (np.isnan(rc) or np.isnan(val)) else val - rc
        if atom == "N":
            try:
                rc = _RC[aa]["N"]
            except KeyError:
                return np.nan
            if np.isnan(rc) or np.isnan(val):
                return np.nan
            base = val - rc
            prev = row["prev_comp_id"]
            if prev is None:
                return base
            corr = _RC_N_PREV.get(prev)
            return base if corr is None else base - (corr - _RC_N_PREV_GLY)
        if atom == "H":
            try:
                rc = _RC[aa]["H"]
            except KeyError:
                return np.nan
            return np.nan if (np.isnan(rc) or np.isnan(val)) else val - rc
        return np.nan

    df["secondary_shift"] = df.apply(_secondary_shift, axis=1)

    csi_by_seq = {}
    for s in seq_ids:
        res = df.loc[df["Seq_ID"] == s]
        ca = res[res["Atom_ID"] == "CA"]
        cb = res[res["Atom_ID"] == "CB"]
        if ca.empty or cb.empty:
            csi_by_seq[s] = np.nan
            continue
        ca_sec = float(ca.iloc[0]["secondary_shift"])
        cb_sec = float(cb.iloc[0]["secondary_shift"])
        csi_by_seq[s] = (
            ca_sec - cb_sec
            if np.isfinite(ca_sec) and np.isfinite(cb_sec) else np.nan
        )
    df["csi"] = df["Seq_ID"].map(csi_by_seq)
    df["csi_prev"] = df["Seq_ID"].map(lambda s: csi_by_seq.get(s - 1, np.nan))

    raw = {}
    for atom in ("CA", "CB"):
        x, y = _prepare_fit_data(df[df["Atom_ID"] == atom], "csi")
        raw[atom] = _piecewise_offset(x, y)

    x, y = _prepare_fit_data(df[df["Atom_ID"] == "C"], "csi")
    mask = ~(np.isnan(x) | np.isnan(y))
    xc, yc = x[mask], y[mask]
    if len(xc) < 20:
        raw["C"] = np.nan
    else:
        _, intercept = _huber_fit(xc, yc)
        raw["C"] = (
            np.nan if intercept is None
            else (0.0 if abs(intercept) <= _THRESH_C else intercept)
        )

    for atom in ("N", "H"):
        expected_slope, slope_tight, threshold = _ATOM_PARAMS[atom]
        x, y = _prepare_fit_data(
            df[df["Atom_ID"] == atom], "csi_prev", exclude_col="prev_comp_id"
        )
        raw[atom] = _lacs_offset_linear(
            x, y, expected_slope, slope_tight, threshold, _SLOPE_TOL[atom]
        )

    # Fit intercept ≈ d_obs − d_ave; flip to offset = d_ave − d_obs.
    offsets, check = {}, {}
    for atom in _LACS_ATOMS:
        v = raw.get(atom)
        ok = v is not None and not (isinstance(v, float) and np.isnan(v))
        check[atom] = ok
        offsets[atom] = -v if ok else None
    return offsets, check
