"""
PANAV re-referencing (Wang & Wishart 2005).

Two-round probabilistic secondary-structure assignment from HA shifts, then
per-atom offsets relative to the SS-specific reference distribution. No
regression — pure gaussian scoring plus robust means. Entry point:
:func:`reref_panav`.
"""

import numpy as np
import pandas as pd

from ..data.tables import get_panav_distns

_PANAV_REF = get_panav_distns()    # {residue: {ss: {atom: (mean, std)}}}

_SS = ("C", "H", "E")              # coil, helix, strand
_NO_REF = {("PRO", "N"), ("PRO", "H"), ("GLY", "CB")}
_PANAV_ATOMS = ("N", "CA", "CB", "C")


def _has_ref(comp_id, atom_id):
    """True if a usable PANAV reference distribution exists for this atom."""
    if (comp_id, atom_id) in _NO_REF:
        return False
    try:
        for ss in _SS:
            mean, std = _PANAV_REF[comp_id][ss][atom_id]
            if pd.isna(mean) or pd.isna(std):
                return False
        return True
    except KeyError:
        return False


def _gaussian(x, mu, sigma):
    if sigma == 0:
        return 0.0
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def _panav_ss_probs(row):
    """Most-probable secondary structure for this shift, from the SS-specific
    reference distributions. Returns (probabilities, ss_label) or (nan, nan)."""
    if not _has_ref(row["Comp_ID"], row["Atom_ID"]):
        return np.nan, np.nan
    scores = [_gaussian(row["Val"], *_PANAV_REF[row["Comp_ID"]][ss][row["Atom_ID"]])
              for ss in _SS]
    if not any(scores):
        return np.nan, np.nan
    selected = _SS[int(np.argmax(scores))]
    if row["Atom_ID"] == "HA":
        mean, std = _PANAV_REF[row["Comp_ID"]][selected][row["Atom_ID"]]
        if abs(row["Val"] - mean) > 4 * std:
            return np.nan, np.nan
    return np.array(scores) / np.sum(scores), selected


def _panav_get_offset(row, corresponding_ha):
    """Offset for this atom relative to the reference mean for the SS assigned
    to the residue's HA. Returns (offset, ss) or (nan, nan)."""
    if not _has_ref(row["Comp_ID"], row["Atom_ID"]) or len(corresponding_ha) == 0:
        return np.nan, np.nan
    ss = corresponding_ha.iloc[0]["ss_max"]
    if pd.isna(ss):
        return np.nan, np.nan
    mean, _ = _PANAV_REF[row["Comp_ID"]][ss][row["Atom_ID"]]
    return row["Val"] - mean, ss


def _panav_judge_outlier(row):
    if not _has_ref(row["Comp_ID"], row["Atom_ID"]):
        return True
    if pd.isna(row["offset"]):
        return all(
            abs(row["Val"] - _PANAV_REF[row["Comp_ID"]][ss][row["Atom_ID"]][0])
            > 4 * _PANAV_REF[row["Comp_ID"]][ss][row["Atom_ID"]][1]
            for ss in _SS
        )
    mean, std = _PANAV_REF[row["Comp_ID"]][row["ss_max"]][row["Atom_ID"]]
    return abs(row["offset"]) > 4 * std


# ── Entry point ──────────────────────────────────────────────────────────────

def reref_panav(df):
    """Compute cumulative per-atom PANAV offsets over two rounds.
    Returns (offsets, check), offsets such that corrected = Val - offset."""
    df = df.copy()
    df["Atom_ID"] = df["Atom_ID"].replace("HA2", "HA")
    df["outlier_1"] = False
    df["outlier_2"] = False

    check = {atom: True for atom in _PANAV_ATOMS}
    cumulative = {atom: 0.0 for atom in _PANAV_ATOMS}

    for round_i in range(2):
        df[["ss_probs", "ss_max"]] = df.apply(
            _panav_ss_probs, axis=1, result_type="expand"
        )
        ha = df.loc[df.Atom_ID == "HA"]
        df[["offset", "ss_max"]] = df.apply(
            lambda r: _panav_get_offset(r, ha.loc[ha.Seq_ID == r["Seq_ID"]]),
            axis=1, result_type="expand",
        )

        outlier_col = f"outlier_{round_i + 1}"
        df[outlier_col] = df.apply(_panav_judge_outlier, axis=1)

        round_offsets = {}
        for atom in _PANAV_ATOMS:
            mask = (df.Atom_ID == atom) & (~df["outlier_1"])
            if round_i == 1:
                mask &= ~df["outlier_2"]
            vals = np.array(df.loc[mask, "offset"], dtype=float)

            if np.isnan(vals).all():
                round_offsets[atom] = None
                check[atom] = False
                continue

            m, s = np.nanmean(vals), np.nanstd(vals)
            vals[(vals > m + 3 * s) | (vals < m - 3 * s)] = np.nan
            if (~np.isnan(vals)).sum() < 25:
                round_offsets[atom] = None
                check[atom] = False
                continue

            round_offsets[atom] = float(np.nanmean(vals))

        apply_now = {a: (0.0 if v is None else v) for a, v in round_offsets.items()}
        df["Val"] = df.apply(
            lambda r: r["Val"] - apply_now.get(r["Atom_ID"], 0.0), axis=1
        )
        for atom in _PANAV_ATOMS:
            if round_offsets.get(atom) is not None:
                cumulative[atom] += round_offsets[atom]

        failed = [a for a, ok in check.items() if not ok]
        df.loc[df["Atom_ID"].isin(failed), outlier_col] = True

    offsets = {atom: (cumulative[atom] if check[atom] else None)
               for atom in _PANAV_ATOMS}
    return offsets, check