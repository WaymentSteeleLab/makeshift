"""
Peak-list alignment and one-to-one matching.

Aligns two peak tables in (H_ppm, N_ppm) — typically a reference assignment
list onto peaks picked from a spectrum.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def _match_hungarian(df, ref_df, tol_h=0.03, tol_n=0.3, h_col="H_ppm", n_col="N_ppm"):
    """
    Match peaks between two DataFrames using the Hungarian algorithm.

    Each peak in ref_df is matched to at most one peak in df within
    (tol_h, tol_n) ppm tolerance.

    Returns a DataFrame aligned to ref_df with added columns: ref_index,
    matched, dist_h, dist_n.
    """
    ref = ref_df.reset_index(drop=True)
    tgt = df.reset_index(drop=True)
    rh = ref[h_col].to_numpy(); rn = ref[n_col].to_numpy()
    th = tgt[h_col].to_numpy(); tn = tgt[n_col].to_numpy()
    dh = (rh[:, None] - th[None, :]) / tol_h
    dn = (rn[:, None] - tn[None, :]) / tol_n
    dist = np.sqrt(dh**2 + dn**2)

    # Augment the cost matrix with "leave unmatched" dummy nodes at a fixed
    # cost of 1.0 tol-unit (i.e. right at the edge of the tolerance window).
    # This lets a single global Hungarian solve prefer "stay unmatched" over
    # a far-away match, so close/overlapping peaks aren't "stolen" by a
    # distant, slightly-better-but-still-bad pairing elsewhere in the
    # spectrum.
    n_ref, n_tgt = dist.shape
    n = n_ref + n_tgt
    UNMATCHED_COST = 1.0
    BIG = 1e6
    cost = np.full((n, n), BIG)
    cost[:n_ref, :n_tgt] = dist
    cost[:n_ref, n_tgt:] = np.eye(n_ref) * UNMATCHED_COST + (1 - np.eye(n_ref)) * BIG
    cost[n_ref:, :n_tgt] = np.eye(n_tgt) * UNMATCHED_COST + (1 - np.eye(n_tgt)) * BIG
    cost[n_ref:, n_tgt:] = 0.0

    ref_pos, tgt_pos = linear_sum_assignment(cost)
    matches = {}
    for rp, tp in zip(ref_pos, tgt_pos):
        if rp < n_ref and tp < n_tgt:
            matches[rp] = tp

    rows = []
    for i, ref_row in ref.iterrows():
        rh_val, rn_val = ref_row[h_col], ref_row[n_col]
        if i in matches:
            j = matches[i]
            row = tgt.iloc[j].to_dict()
            row.update(ref_index=i, matched=True,
                       dist_h=float(tgt.iloc[j][h_col] - rh_val),
                       dist_n=float(tgt.iloc[j][n_col] - rn_val))
        else:
            row = {col: np.nan for col in tgt.columns}
            row.update(ref_index=i, matched=False, dist_h=np.nan, dist_n=np.nan)
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def _find_offset(left, right, tol):
    """Grid search for the (Δ¹H, Δ¹⁵N) that minimizes the RMSE of
    (tol-normalized) nearest-neighbor distance from right to left.

    Returns (best_dh, best_dn).
    """
    tol_h, tol_n = tol
    spec_H = left['H_ppm'].values
    spec_N = left['N_ppm'].values
    bH     = right['H_ppm'].values
    bN     = right['N_ppm'].values

    print('  Running grid search for best (Δ¹H, Δ¹⁵N) offset...')
    diff_H = bH[:, None] - spec_H[None, :]   # (n_bmrb, n_spec)
    diff_N = bN[:, None] - spec_N[None, :]
    dh_vals = np.arange(-2.0,  2.0  + 0.01, 0.01)
    dn_vals = np.arange(-10.0, 10.0 + 0.1,  0.05)
    # Score each offset by RMSE of (normalized) nearest-spectrum-peak
    # distance, averaged over peaklist peaks. Normalizing each axis by
    # its tolerance gives H and N equal weight in the RMSE regardless
    # of their different ppm scales. A peaklist peak with no spectrum
    # peak within tolerance is capped at the tolerance edge (1.0) so a
    # few outliers don't dominate the score.
    best_score, best_dh, best_dn = np.inf, 0.0, 0.0
    for dh in dh_vals:
        h_norm = (diff_H + dh) / tol_h
        for dn in dn_vals:
            n_norm = (diff_N + dn) / tol_n
            dist_norm = np.sqrt(h_norm**2 + n_norm**2)
            nearest = np.clip(dist_norm.min(axis=1), 0.0, 1.0)
            score = float(np.sqrt(np.mean(nearest**2)))
            if score < best_score:
                best_score, best_dh, best_dn = score, float(dh), float(dn)
    h_norm = (diff_H + best_dh) / tol_h
    n_norm = (diff_N + best_dn) / tol_n
    dist_norm = np.sqrt(h_norm**2 + n_norm**2)
    n_within = int((dist_norm.min(axis=1) < 1.0).sum())
    print(f'  Best offset: Δ¹H = {best_dh:+.3f}, Δ¹⁵N = {best_dn:+.3f} ppm '
          f'(RMSE = {best_score:.3f}, {n_within}/{len(right)} peaks within tolerance)')
    return best_dh, best_dn


def _detect_conflicts(right_mapped, left, matched, tol, tmp_label):
    """Flag right-side peaks that were within tolerance of a left-side peak
    that got assigned to a closer competitor.

    Returns the `matched` DataFrame with a 'conflict' column added.
    """
    tol_h, tol_n = tol
    spec_H = left['H_ppm'].values
    spec_N = left['N_ppm'].values

    assigned_labels   = set(matched.loc[matched['matched'], tmp_label].dropna())
    conflict_spec_idx = set()
    conflict_info     = []

    for _, brow in right_mapped.iterrows():
        if brow[tmp_label] in assigned_labels:
            continue
        dh_vec = np.abs(brow['H_ppm'] - spec_H)
        dn_vec = np.abs(brow['N_ppm'] - spec_N)
        in_tol = (dh_vec < tol_h) & (dn_vec < tol_n)
        if in_tol.any():
            dist_arr = np.sqrt((dh_vec / tol_h)**2 + (dn_vec / tol_n)**2)
            dist_arr[~in_tol] = np.inf
            spec_idx = int(np.argmin(dist_arr))
            conflict_spec_idx.add(spec_idx)
            conflict_info.append(dict(
                spectrum_peak=spec_idx,
                winner=matched.loc[spec_idx, tmp_label],
                loser=brow[tmp_label],
            ))

    matched['conflict'] = matched['ref_index'].isin(conflict_spec_idx)

    if conflict_info:
        print(f'  {len(conflict_info)} conflict(s) flagged for manual inspection:')
        for c in conflict_info:
            print(f'    spectrum peak {c["spectrum_peak"]:3d}: '
                  f'assigned {c["winner"]},  competing {c["loser"]}')
    else:
        print('  No conflicts.')

    return matched


def _apply_how(left, right_mapped, matched, how, right_label, tmp_label):
    """
    Build (left_out, right_mapped) according to `how`, mirroring pandas
    merge semantics. See map_peaklists for the meaning of each mode.
    """
    if how not in ("left", "right", "inner", "outer"):
        raise ValueError(f"how must be one of 'left', 'right', 'inner', 'outer', got {how!r}")

    out = left.copy()
    if right_label is not None:
        out[right_label] = matched[tmp_label].values
    out['conflict'] = matched['conflict'].values

    if how in ("inner", "right"):
        out = out[matched['matched'].values].reset_index(drop=True)

    # for "right"/"outer", append rows for right-side peaks with no left match
    if how in ("right", "outer"):
        matched_right_vals = set(matched.loc[matched['matched'], tmp_label])
        leftover = right_mapped[~right_mapped[tmp_label].isin(matched_right_vals)]
        if len(leftover):
            extra = pd.DataFrame(index=leftover.index, columns=out.columns)
            extra['H_ppm'] = leftover['H_ppm'].values
            extra['N_ppm'] = leftover['N_ppm'].values
            if right_label is not None:
                extra[right_label] = leftover[tmp_label].values
            extra['conflict'] = False
            out = pd.concat([out, extra], ignore_index=True)

    if how in ("inner", "right"):
        out['ref_index'] = out.index

    return out, right_mapped


def map_peaklists(left, right, offset=None,
                  tol=1, label_cols=("assn_label", "assn_label"),
                  how="inner"):
    """
    Align two peak lists in (H_ppm, N_ppm) and match them one-to-one.

    Typical use is mapping a reference peaklist containing assignments (right)
    onto peaks picked from a spectrum (left), but either side can be any
    DataFrame with H_ppm/N_ppm columns — e.g. matching peaks picked from two
    different spectra to each other. `right` is the side that gets shifted by
    the offset; `left` is treated as fixed.

    Step 1 — Translation offset: if offset is None, a grid search finds the
    (Δ1H, Δ15N) that minimizes the RMSE of nearest-neighbor distance
    (normalized by (tol_h, tol_n) so both dimensions contribute equally)
    between right and left.

    Step 2 — One-to-one Hungarian matching within the tolerance window.

    Step 3 — Conflict detection: right-side peaks that were within tolerance
    of a left-side peak that got assigned to a closer competitor are flagged.

    Parameters
    ----------
    left : DataFrame
        Picked spectrum peaks with H_ppm and N_ppm columns. Treated as fixed.
    right : DataFrame
        Peaks to map onto left, with H_ppm, N_ppm, and a label column (e.g. a
        PeakList's .data). Shifted by the offset before matching.
    offset : (float, float) or None
        (Δ¹H, Δ¹⁵N) to apply to `right` before matching. If None, a grid
        search finds the best offset.
    tol : float
        Scalar multiplier on the default post-shift matching tolerances
        (tol_h, tol_n) = (0.03, 0.3) * tol, in ppm.
    label_cols : (str, str)
        Label column names on (left, right) used for conflict detection and
        carried through to the outputs. Either can be None if that side has
        no label column.
    how : {"inner", "left", "right", "outer"}
        Mirrors pandas merge semantics, and applies only to left_out — the
        returned right_mapped always contains the full right peaklist
        (translated by the offset), regardless of `how`.
        "inner" (default): left_out contains only matched left peaks.
        "right": matched left peaks, plus one extra row per unmatched right
        peak (left-side columns NaN).
        "left": all left-side peaks, matched or not.
        "outer": union — all left-side peaks plus one extra row per unmatched
        right peak (left-side columns NaN).

    Returns
    -------
    left_out : DataFrame
        Updated with <right label_col> and conflict columns, filtered/extended
        according to `how`.
    right_mapped : DataFrame
        Full right-side peaklist after offset (for plotting), independent of
        `how`.
    """
    left_label, right_label = label_cols
    tol_h, tol_n = 0.03 * tol, 0.3 * tol

    #  offset 
    if offset is not None:
        best_dh, best_dn = float(offset[0]), float(offset[1])
        print(f'  Using provided offset: Δ¹H = {best_dh:+.3f}, '
              f'Δ¹⁵N = {best_dn:+.3f} ppm')
    else:
        best_dh, best_dn = _find_offset(left, right, (tol_h, tol_n))

    #  shift and match 
    right_mapped = right.copy()
    right_mapped['H_ppm'] += best_dh
    right_mapped['N_ppm'] += best_dn

    # use a temporary label column internally if the right side has none
    tmp_label = right_label or '_right_index'
    if right_label is None:
        right_mapped[tmp_label] = right_mapped.index

    matched = _match_hungarian(right_mapped, left,
                               tol_h=tol_h, tol_n=tol_n,
                               h_col='H_ppm', n_col='N_ppm')

    #  conflict detection 
    matched = _detect_conflicts(right_mapped, left, matched, (tol_h, tol_n), tmp_label)

    #  build updated left 
    out, right_mapped = _apply_how(left, right_mapped, matched, how, right_label, tmp_label)

    if right_label is None:
        right_mapped = right_mapped.drop(columns=[tmp_label])

    n_assigned = matched['matched'].sum()
    n_conflict = matched['conflict'].sum()
    print(f'  {n_assigned}/{len(right)} peaks in right peaklist assigned  |  '
          f'{n_conflict} flagged for inspection')

    return out, right_mapped