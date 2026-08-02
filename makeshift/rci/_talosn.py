"""
TALOS-N's own RCI-S2 module (``RCI.cpp``/``TALOS.cpp``, v4.11 source).

This is a different algorithm from the ``rci_v_1c.py`` script ported in
:mod:`makeshift.rci.engine` — TALOS-N bundles its own simplified
reimplementation, which has drifted from the original. The point of this
port is to match what the compiled binary actually produces, so its quirks
are reproduced rather than corrected:

    no early floor on raw deviations
    strict +-1 window averaging, no gap filling
    calcRCI divides by a constant 5, not the number of atoms present
    Gly counts as missing a CB on top of whatever CB value is used
    applyEndCorrection scans residues 1-4 by absolute number
    unobserved atoms get a synthesized deviation, not zero
    Gly HA2/HA3 are averaged into one HA before use

Not ported: TALOS.cpp's ``calcAverageCS`` which predicts an isolated missing 
shift by searching TALOS-N's bundled reference protein database,
falling back to the table residual used here only when that search fails.
Porting it means porting the database and its scoring too. This is the
dominant remaining difference from the binary, concentrated at Pro and Gly;
``docs/rci_validation.md`` has the numbers.
"""

import math

import numpy as np
import pandas as pd

from ..data.tables import get_talosn_rc_tables

# AvgCSMatrix / inputCSMatrix order in RCI.cpp. HN is carried through but
# drops out of the sum, since its weight is zero.
_ATOM_ORDER = ["HN", "N", "HA", "C", "CA", "CB"]
_ATOM_TO_MAKESHIFT = {"HN": "H", "N": "N", "HA": "HA", "C": "C", "CA": "CA", "CB": "CB"}
# talosn_*.csv columns -> makeshift atom names (same [N,CO,CA,CB,NH,HA]
# convention as RCI's own tables).
_TABLE_ATOM_TO_MAKESHIFT = {"N": "N", "CO": "C", "CA": "CA", "CB": "CB", "NH": "H", "HA": "HA"}
_HERTZ = {"HN": 10.0, "N": 1.0, "HA": 10.0, "C": 2.5, "CA": 2.5, "CB": 2.5}
_UWEIGHT = {"HN": 0.0, "N": 0.59, "HA": 0.85, "C": 0.72, "CA": 0.72, "CB": 0.15}
_FLOOR = 0.5
_CEILING = 0.6
_SCALE = 1.125
_OXIDIZED_CYS_CB_PPM = 35.0
_OXIDIZED_CYS_CB_PPM_TALOSN = 34.0
_S2_A = 0.4
_S2_B = 17.7
_S2_OFFSET = 1.003


def _talosn_rc_reference(seq_map, shifts):
    """
    ``talosCS_RC``: TALOS-N's own random coil reference, built as
    ``randcoil + rcadj[self] + rcprev[prev] + rcnext[next]``.

    A separate, simpler table system from `simpred` — only +-1 neighbors,
    and its own oxidized-Cys threshold of 34.0 ppm rather than RCI's 35.0.

    An atom that was never observed doesn't simply drop out: TALOS-N gives
    it a deviation of ``talosCS_RC - simpred``, which is usually nonzero and
    sequence dependent, and which the window average below has no way to
    tell apart from real signal. It matters most for sparse depositions
    (N/H only), where there is little real signal to average it against.
    """
    tables = get_talosn_rc_tables()
    randcoil, rcadj, rcprev, rcnext = (
        tables["randcoil"], tables["rcadj"], tables["rcprev"], tables["rcnext"]
    )
    cols = list(randcoil.columns)

    oxidized_cys = set(
        shifts.loc[
            (shifts["Comp_ID"].str.upper() == "CYS")
            & (shifts["Atom_ID"] == "CB")
            & (shifts["Val"] >= _OXIDIZED_CYS_CB_PPM_TALOSN),
            "Seq_ID",
        ].astype(int)
    )

    def code_of(r):
        aa = seq_map.get(r)
        if aa == "C" and r in oxidized_cys:
            return "c"
        return aa

    def lookup(table, code):
        if code is None or code not in table.index:
            return pd.Series(0.0, index=cols)
        return table.loc[code]

    rows = {}
    for r in seq_map:
        self_code = code_of(r)
        rows[r] = (
            lookup(randcoil, self_code)
            + lookup(rcadj, self_code)
            + lookup(rcprev, code_of(r - 1))
            + lookup(rcnext, code_of(r + 1))
        )
    df = pd.DataFrame.from_dict(rows, orient="index")[cols]
    df = df.rename(columns=_TABLE_ATOM_TO_MAKESHIFT)
    df.index.name = "Seq_ID"
    return df.sort_index()


def run_talosn_rci(shifts, seq_map, simpred):
    """
    RCI and S2 as RCI.cpp + TALOS.cpp compute them.

    `simpred` is the same reference table the ``rci_v_1c.py`` port uses
    (:func:`makeshift.rci.engine._build_simpred`) — TALOS-N's compiled-in
    tables are byte-identical to the Schwarzinger tables shipped here.
    """
    first_residue, last_residue = min(seq_map), max(seq_map)
    all_residues = list(range(first_residue, last_residue + 1))

    oxidized_cys = set(
        shifts.loc[
            (shifts["Comp_ID"].str.upper() == "CYS")
            & (shifts["Atom_ID"] == "CB")
            & (shifts["Val"] > _OXIDIZED_CYS_CB_PPM),
            "Seq_ID",
        ].astype(int)
    )

    talosn_rc = _talosn_rc_reference(seq_map, shifts)

    # inCS_convert_TALOS2RCI: raw deviations, no early floor, plus a
    # missing-ness tag tracked separately from the value. Pro N/HN and Gly
    # CB have no reference so their value is 0.0, but that alone doesn't
    # make them "missing". Atoms never observed get the synthesized
    # deviation described in _talosn_rc_reference.
    raw_dev = {}
    observed_present = {}
    for atom in _ATOM_ORDER:
        makeshift_atom = _ATOM_TO_MAKESHIFT[atom]
        obs = shifts[shifts["Atom_ID"].str[:2] == makeshift_atom]
        if atom == "CB":
            obs = obs[~obs["Seq_ID"].isin(oxidized_cys)]
        observed = {}
        for r in obs.itertuples():
            if pd.notna(r.Val):
                observed.setdefault(int(r.Seq_ID), []).append(float(r.Val))
        simpred_col = simpred[makeshift_atom]
        talosn_rc_col = talosn_rc[makeshift_atom]
        d = {}
        present = {}
        for r in all_residues:
            if r in observed:
                present[r] = True
            sp = simpred_col.get(r)
            if sp is None or (isinstance(sp, float) and np.isnan(sp)):
                d[r] = 0.0
                continue
            if r in observed:
                vals = observed[r]
                # TALOS.cpp averages Gly's HA2/HA3 into a single HA. Guard
                # this tightly: a deposition with two chains also puts >1 row
                # on a slot, for every atom type, and those should not be
                # averaged — take the first, as TALOS-N does.
                if atom == "HA" and seq_map.get(r) == "G" and len(vals) > 1:
                    obs_val = sum(vals) / len(vals)
                else:
                    obs_val = vals[0]
                d[r] = obs_val - sp
            else:
                trc = talosn_rc_col.get(r)
                if trc is None or (isinstance(trc, float) and np.isnan(trc)):
                    continue
                d[r] = trc - sp
        raw_dev[atom] = d
        observed_present[atom] = present

    def window_avg(atom, r, lo, hi):
        vals = []
        for k in range(lo, hi + 1):
            v = raw_dev[atom].get(r + k)
            if v is not None and v != 0:
                vals.append(abs(v))
        return sum(vals) / len(vals) if vals else 0.0

    avg_cs = {}
    for r in all_residues:
        avg_cs[r] = {}
        for atom in _ATOM_ORDER:
            if r == first_residue:
                avg_cs[r][atom] = window_avg(atom, r, 0, 1)
            elif r == last_residue:
                avg_cs[r][atom] = window_avg(atom, r, -1, 0)
            else:
                avg_cs[r][atom] = window_avg(atom, r, -1, 1)

    miss_count = {}
    for r in all_residues:
        cnt = sum(1 for atom in ["N", "HA", "C", "CA", "CB"] if not observed_present[atom].get(r, False))
        if seq_map[r] == "G":
            cnt += 1
        miss_count[r] = cnt

    input_cs = {}
    for r in all_residues:
        input_cs[r] = {}
        for atom in _ATOM_ORDER:
            v = avg_cs[r][atom] * _HERTZ[atom]
            if abs(v) < _FLOOR:
                v = _FLOOR
            input_cs[r][atom] = v

    output_rci = {}
    for r in all_residues:
        s = sum(abs(input_cs[r][atom]) * _UWEIGHT[atom] * 5.0 for atom in ["N", "HA", "C", "CA", "CB"])
        s /= 5.0
        rci = (1.0 / s) ** 1.5 if s != 0 else float("inf")
        if rci > _CEILING:
            rci = _CEILING
        # note the == : a Gly missing everything counts 6, not 5
        if miss_count[r] == 5:
            rci = 0.0
        output_rci[r] = rci / _SCALE

    def end_correction(values):
        """
        applyEndCorrection: pull the terminal residues up toward the local
        maximum. The N-terminal scan walks residues 1-4 by absolute number,
        so it does nothing useful for a chain that starts higher up. Note
        the ceiling is applied here in scaled units, after the divide above,
        unlike the wishart path which caps before scaling.
        """
        result = dict(values)
        max_pos, max_rci = 1, -1.0
        for i in range(1, 5):
            # skip residues below the chain start, and the degenerate case
            # where the chain starts exactly at 4 and there is nothing left
            # of it to correct
            if i < first_residue or (i == first_residue and i == 4):
                continue
            if i not in miss_count:
                continue
            if miss_count[i] < 5 and result[i] > max_rci:
                max_rci, max_pos = result[i], i
        if max_rci > -1:
            for i in range(max_pos - 1, 0, -1):
                if i in miss_count and miss_count[i] < 5:
                    diff = max_rci - result[i]
                    result[i] = min(result[i] + diff * 2.0, _CEILING)

        max_pos, max_rci = last_residue, -1.0
        for i in range(last_residue - 3, last_residue + 1):
            if i not in miss_count:
                continue
            if miss_count[i] < 5 and result[i] > max_rci:
                max_rci, max_pos = result[i], i
        for i in range(max_pos + 1, last_residue + 1):
            if i in miss_count and miss_count[i] < 5:
                diff = max_rci - result[i]
                result[i] = min(result[i] + diff * 2.0, _CEILING)
        return result

    output_rci = end_correction(output_rci)

    def final_smooth(values):
        """
        +-1 smoothing over residues that have data. A residue with no
        usable neighbors gets 9999, TALOS-N's no-data marker in predS2.tab,
        which is passed through to the results table as-is.
        """
        result = {}
        def smoothed(r, lo, hi):
            vals = [values[r + k] for k in range(lo, hi + 1)
                    if (r + k) in miss_count and miss_count[r + k] < 5]
            return sum(vals) / len(vals) if vals else 9999.0
        for r in all_residues:
            if r == first_residue:
                result[r] = smoothed(r, 0, 1)
            elif r == last_residue:
                result[r] = smoothed(r, -1, 0)
            else:
                result[r] = smoothed(r, -1, 1)
        return result

    output_rci = final_smooth(output_rci)

    rows = []
    for r in all_residues:
        rci = output_rci[r]
        if rci == 9999.0:
            s2 = 9999.0
        else:
            s2 = _S2_OFFSET - _S2_A * math.log(1.0 + rci * _S2_B)
        rows.append({"Seq_ID": r, "Comp_ID": seq_map.get(r), "RCI": rci, "S2": s2})
    return pd.DataFrame(rows)
