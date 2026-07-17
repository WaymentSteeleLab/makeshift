"""
Exact port of TALOS-N's own RCI-S2 algorithm (``RCI.cpp``/``TALOS.cpp`` in
the TALOS-N v4.11 source), as opposed to the ``rci_v_1c.py`` script that
:mod:`makeshift.rci.engine` otherwise ports. TALOS-N bundles a simplified,
independently-drifted reimplementation of Wishart's RCI with several bugs
relative to the original script, all reproduced faithfully here since the
goal is to match TALOS-N's actual output, not to fix it:

- No early-floor (0.1/Hz) clamp on raw deviations before averaging.
- Simple strict +-1 residue window averaging (no gap-filling/borrowing).
- ``calcRCI``'s denominator is a constant 5, not the count of atoms with
  data actually present.
- Gly always counts as "missing" for the CB slot in ``missCSCount``, on
  top of (independently of) whatever value is used for it.
- ``applyEndCorrection``'s N-terminal window uses hardcoded absolute
  residue numbers 1-4, not relative to the chain's actual first residue.
- S2 is written to file as ``1.003 - 0.4*ln(1 + 17.7*RCI)``: the internal
  ``0.4*ln(1+17.7*RCI)`` value (which increases with RCI, i.e. is *not*
  in the conventional S2 sense) gets inverted by TALOS.cpp before output.
- An atom that was never observed at all is *not* simply treated as
  missing data: ``inCS_convert_TALOS2RCI`` synthesizes a deviation for it
  from TALOS-N's own random-coil-adjustment tables (``randcoil.tab``/
  ``rcadj.tab``/``rcprev.tab``/``rcnext.tab``, ported to
  ``makeshift/data/rci_data/talosn_*.csv`` and read via
  :func:`makeshift.data.tables.get_talosn_rc_tables`) minus RCI's own
  Schwarzinger-table reference -- a real, usually-nonzero, sequence-
  dependent number that isn't filtered out downstream (the window-average
  only skips exact zeros). This matters most for inputs missing most of
  HA/C/CA/CB (e.g. N/H-only depositions), where it dominates instead of
  averaging away against plentiful real signal; see
  :func:`_talosn_rc_reference`.
- Gly's HA2/HA3 are averaged into one HA value before use (TALOS.cpp's
  in2_Tab special-case for resName=="G" && atom=="HA"), not just the
  first-matched row.

Known remaining gap (not yet ported): when a residue has only 1-2 missing
atoms and its immediate neighbors are also mostly complete, TALOS.cpp's
``calcAverageCS`` (called from the ANN-input-prep stage, not RCI.cpp
itself) tries to *predict* a value for the missing atom via a homology-
weighted nearest-neighbor search against TALOS-N's bundled reference
protein database, and only falls back to the plain
``talosn_rc - simpred`` residual above if that search fails
(``testTriplet`` requires the residue and its neighbors to already be
fairly complete, so this essentially never engages for sparse N/H-only
inputs -- where the plain residual is exactly right -- but does engage
for isolated missing atoms in otherwise-dense depositions). Porting
``calcAverageCS`` would mean porting that reference database and its
search/scoring algorithm too; not attempted. This is the dominant source
of the remaining error, concentrated in Pro/Gly and their immediate
neighbors (mean abs S2 diff by residue type: Pro 0.033, Gly 0.018, vs.
0.005-0.016 for everything else).

Confirmed against a debug-instrumented rebuild of the actual v4.11 source
under Docker (matching real ``predS2.tab`` output) and validated across 9
BMRB entries against the real compiled TALOS-N binary (pooled r=0.989,
median abs S2 diff=0.0019, 78% of residues within 0.01, 96% within 0.05).
"""

import math

import numpy as np
import pandas as pd

from ..data.tables import get_talosn_rc_tables

# AvgCSMatrix / inputCSMatrix order in RCI.cpp; HN is tracked but excluded
# from calcRCI's sum (uWeight["HN"] == 0.0).
_ATOM_ORDER = ["HN", "N", "HA", "C", "CA", "CB"]
_ATOM_TO_MAKESHIFT = {"HN": "H", "N": "N", "HA": "HA", "C": "C", "CA": "CA", "CB": "CB"}
# talosn_*.csv column names (the same [N,CO,CA,CB,NH,HA] convention as RCI's
# own wide tables) -> makeshift atom names.
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
    """``talosCS_RC`` in TALOS.cpp: TALOS-N's own random-coil-adjustment
    reference (``randcoil + rcadj[self] + rcprev[prev] + rcnext[next]``).

    This is a separate, simpler table system from RCI's own
    ``rciCS_RC``/`simpred` (only +-1 neighbor, no +-2), with its own
    oxidized-Cys code swap at TALOS-N's own >=34.0ppm CB threshold
    (``RC_Tab``/``ADJ_Tab``/``PREV_Tab``/``NEXT_Tab`` in TALOS.cpp,
    populated from ``randcoil.tab``/``rcadj.tab``/``rcprev.tab``/
    ``rcnext.tab`` -- distinct from RCI's 35.0ppm threshold used for
    `oxidized_cys` in :func:`run_talosn_rci`). ``inCS_convert_TALOS2RCI``
    uses this value to synthesize a nonzero deviation for atoms that were
    never observed at all: ``talosCS_RC[atom] - rciCS_RC[atom]``, which
    (unlike an actually-missing atom) does *not* get skipped by the
    downstream window-averaging's "value != 0" check.
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
    """Compute RCI/S2 exactly as TALOS-N's own RCI.cpp + TALOS.cpp do.

    `simpred` is the same random-coil-plus-neighbor-correction reference
    table used by the ``rci_v_1c.py`` port (:func:`makeshift.rci.engine.
    _build_simpred`); TALOS-N's compiled-in tables were confirmed
    byte-identical to the Schwarzinger tables this repo ships.
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

    # inCS_convert_TALOS2RCI: raw deviation (no early floor) + a
    # missing-ness tag tracked independently of whether a random-coil
    # reference exists (Pro N/HN, Gly CB force the *value* to 0.0 but do
    # not themselves make the atom "missing"). An atom that was never
    # observed at all is *not* simply zeroed out -- it gets a synthetic
    # deviation from TALOS-N's own random-coil reference
    # (`talosn_rc[atom] - simpred[atom]`), which the window-averaging
    # below treats as real signal unless it happens to be exactly zero.
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
                if atom == "HA" and seq_map.get(r) == "G" and len(vals) > 1:
                    # TALOS.cpp explicitly averages Gly's HA2/HA3 into one
                    # HA value (in2_Tab special-case for resName=="G" &&
                    # atom=="HA") -- but some depositions carry duplicate
                    # full assignment sets (e.g. two chains/entities), which
                    # also land >1 row on the same (residue, atom) slot for
                    # every atom type, not just Gly HA2/HA3; only average
                    # when it's actually the Gly case TALOS-N's code covers,
                    # not any incidental duplicate.
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
        if miss_count[r] == 5:
            rci = 0.0
        output_rci[r] = rci / _SCALE

    def end_correction(values):
        result = dict(values)
        max_pos, max_rci = 1, -1.0
        for i in range(1, 5):
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
