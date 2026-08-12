"""
Classic Wishart Chemical-Shift Index (CSI).

References
----------
Wishart, Sykes & Richards (1992) Biochemistry 31:1647-1651 — 1Ha CSI.
Wishart & Sykes (1994) J. Biomol. NMR 4:171-180 — 13Ca / 13Cβ / 13C' CSI
and consensus secondary structure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data.tables import get_csi_wishart

# Nucleus → (helix ternary sign, strand ternary sign, strand_only)
# HA: upfield (−) = helix, downfield (+) = strand
# CA / C′: downfield (+) = helix, upfield (−) = strand
# CB: downfield (+) = strand only (helix identification unreliable)
_NUCLEUS_SIGNS = {
    "HA": (-1, 1, False),
    "CA": (1, -1, False),
    "C": (1, -1, False),
    "CB": (None, 1, True),
}

_WISHART_ATOMS = ("HA", "CA", "CB", "C")


def ternary_index(val, center, half_width):
    """Return −1 / 0 / +1 for a shift vs. Wishart center ± half-width range."""
    if val is None or (isinstance(val, float) and np.isnan(val)) or np.isnan(center):
        return np.nan
    lo, hi = center - half_width, center + half_width
    if val > hi:
        return 1.0
    if val < lo:
        return -1.0
    return 0.0


def _lookup_range(comp_id, atom, tables, cb_val=None):
    """Return (center, half_width) for residue/atom; handle Cys ox/red."""
    aa = str(comp_id).upper()
    atom = str(atom).upper()
    if aa == "CYS":
        # Prefer oxidized table when CB is nearer the disulfide reference.
        if cb_val is not None and np.isfinite(cb_val):
            red = tables.get("CYS", {}).get("CB")
            ox = tables.get("CYSO", {}).get("CB")
            if red and ox:
                if abs(cb_val - ox[0]) < abs(cb_val - red[0]):
                    aa = "CYSO"
        key = tables.get(aa, {}).get(atom)
        if key is None and aa == "CYSO":
            key = tables.get("CYS", {}).get(atom)
        return key if key is not None else (np.nan, np.nan)
    key = tables.get(aa, {}).get(atom)
    return key if key is not None else (np.nan, np.nan)


def _ha_value(res_rows):
    """Observed Hα; average Gly HA2/HA3 when both present."""
    ha = res_rows[res_rows["Atom_ID"] == "HA"]
    if len(ha):
        return float(ha.iloc[0]["Val"])
    gly = res_rows[res_rows["Atom_ID"].isin(("HA2", "HA3"))]
    vals = gly["Val"].dropna()
    if len(vals):
        return float(vals.mean())
    return np.nan


def _atom_value(res_rows, atom):
    if atom == "HA":
        return _ha_value(res_rows)
    hit = res_rows[res_rows["Atom_ID"] == atom]
    if not len(hit):
        return np.nan
    return float(hit.iloc[0]["Val"])


def residue_indices(data, atoms=_WISHART_ATOMS):
    """
    Per-residue Wishart ternary indices.

    Returns a DataFrame with columns Seq_ID, Comp_ID, and one column per
    requested atom (HA / CA / CB / C). Glycine's missing CB is proxied by
    its HA index (Wishart & Sykes 1994, Table 2 footnote).
    """
    tables = get_csi_wishart()
    atoms = tuple(a.upper() for a in atoms)
    rows = []
    for seq_id, res in data.groupby("Seq_ID", sort=True):
        comp = str(res.iloc[0]["Comp_ID"]).upper()
        cb_val = _atom_value(res, "CB")
        entry = {"Seq_ID": int(seq_id), "Comp_ID": comp}
        for atom in atoms:
            val = _atom_value(res, atom)
            center, hw = _lookup_range(comp, atom, tables, cb_val=cb_val)
            entry[atom] = ternary_index(val, center, hw)
            entry[f"{atom}_raw"] = val - center if np.isfinite(val) and np.isfinite(center) else np.nan
        # Gly has no CB — proxy with HA CSI
        if "CB" in atoms and comp == "GLY" and not np.isfinite(entry.get("CB", np.nan)):
            entry["CB"] = entry.get("HA", np.nan)
            entry["CB_raw"] = entry.get("HA_raw", np.nan)
        rows.append(entry)
    return pd.DataFrame(rows)


def assign_secondary_structure(indices, nucleus="CA"):
    """
    Stage-2 density filter: ternary CSI → H / E / C along the sequence.

    Implements Wishart et al. (1992) steps 3-5 and Wishart & Sykes (1994)
    steps 3-8 (CB is strand-only).
    """
    nucleus = nucleus.upper()
    if nucleus not in _NUCLEUS_SIGNS:
        raise ValueError(f"unknown CSI nucleus {nucleus!r}")
    helix_sign, strand_sign, strand_only = _NUCLEUS_SIGNS[nucleus]

    vals = []
    for v in indices:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            vals.append(0)  # missing treated as coil for segmentation
        else:
            vals.append(int(v))
    n = len(vals)
    ss = ["C"] * n

    if not strand_only and helix_sign is not None:
        for i, j in _segments(vals, helix_sign, -helix_sign,
                              min_count=4, min_consecutive=None, density=0.7):
            for k in range(i, j):
                ss[k] = "H"

    for i, j in _segments(vals, strand_sign, -strand_sign,
                          min_count=4 if strand_only else 3,
                          min_consecutive=3, density=0.7):
        for k in range(i, j):
            # helix wins if already assigned (shouldn't overlap often)
            if ss[k] == "C":
                ss[k] = "E"

    return ss


def _segments(vals, target, opposite, min_count, min_consecutive, density):
    """Yield (start, end) half-open spans that pass Wishart density rules."""
    n = len(vals)
    i = 0
    while i < n:
        if vals[i] == opposite:
            i += 1
            continue
        j = i
        zero_run = 0
        while j < n:
            v = vals[j]
            if v == opposite:
                break
            if v == 0:
                zero_run += 1
                if zero_run >= 2 and j > i:
                    break  # two consecutive zeros terminate (exclude them)
            else:
                zero_run = 0
            j += 1
        end = j
        while end > i and vals[end - 1] == 0:
            end -= 1
        if end > i:
            seg = vals[i:end]
            n_target = sum(1 for v in seg if v == target)
            length = len(seg)
            n_nz = sum(1 for v in seg if v != 0)
            ok_density = (n_nz / length) >= density if length else False
            ok_count = n_target >= min_count
            ok_consec = True
            if min_consecutive is not None:
                run = best = 0
                for v in seg:
                    if v == target:
                        run += 1
                        best = max(best, run)
                    else:
                        run = 0
                ok_consec = best >= min_consecutive
            # also require ≥70% of a 4–5 residue window somewhere — overall
            # segment density covers the usual case for short elements
            if ok_count and ok_density and ok_consec:
                yield i, end
        i = max(end, i + 1) if end > i else i + 1


def consensus_ss(ss_by_nucleus, min_voters=3):
    """
    Majority-rules consensus (Wishart & Sykes 1994): need ≥ ``min_voters``
    nuclei; ties → coil.
    """
    nuclei = list(ss_by_nucleus)
    if not nuclei:
        return []
    n = len(ss_by_nucleus[nuclei[0]])
    out = []
    for i in range(n):
        votes = [ss_by_nucleus[nuc][i] for nuc in nuclei
                 if ss_by_nucleus[nuc][i] in ("H", "E", "C")]
        # CB "C" means non-strand; still a vote for coil
        if len(votes) < min_voters:
            out.append("C")
            continue
        counts = {"H": 0, "E": 0, "C": 0}
        for v in votes:
            counts[v] += 1
        # majority: more than half of available votes
        winner = max(counts, key=counts.get)
        if counts[winner] * 2 > len(votes) and counts[winner] >= 2:
            out.append(winner)
        else:
            out.append("C")
    return out


def wishart_table(data, atoms=_WISHART_ATOMS, assign_ss=True):
    """
    Full Wishart CSI table: ternary indices, optional per-nucleus SS, consensus.

    Returns one row per ``Seq_ID``.
    """
    idx = residue_indices(data, atoms=atoms)
    if not assign_ss:
        return idx

    ss_map = {}
    for atom in atoms:
        if atom not in idx.columns:
            continue
        series = idx[atom]
        # Skip nuclei with no observed shifts — an all-missing series would
        # otherwise vote "coil" everywhere and spoil the consensus.
        if not series.notna().any():
            continue
        ss_map[atom] = assign_secondary_structure(series.tolist(), nucleus=atom)
        idx[f"ss_{atom}"] = ss_map[atom]

    if len(ss_map) >= 3:
        idx["ss"] = consensus_ss(ss_map, min_voters=3)
    elif len(ss_map) == 1:
        only = next(iter(ss_map))
        idx["ss"] = ss_map[only]
    elif len(ss_map) == 2:
        # not enough for paper consensus — leave coil / use agreement
        a, b = list(ss_map)
        idx["ss"] = [
            ss_map[a][i] if ss_map[a][i] == ss_map[b][i] else "C"
            for i in range(len(idx))
        ]
    else:
        idx["ss"] = ["C"] * len(idx)

    # ternary view of consensus for plotting / column parity with LACS CSI
    idx["csi"] = [{"H": 1.0, "E": -1.0, "C": 0.0}[s] for s in idx["ss"]]
    return idx
