"""
RCI (Random Coil Index) engine: a pure-Python/pandas port of the flexibility
predictor from assigned backbone chemical shifts (Berjanskii & Wishart 2005),
based on the reference implementation ``rci_v_1c.py``.

This ports exactly the code path that script takes with its own default
settings (Schwarzinger random-coil table, Schwarzinger neighbor corrections,
3-point smoothing, ``end_effect3`` termini correction, ``function_flag==8``
sigma combination) -- the ~30 alternate branches behind its other CLI flags
are not reproduced. The one deliberate deviation is ``S2``: the published
Berjanskii & Wishart 2005 relation is used instead of ``rci_v_1c.py``'s own
``.S2.txt`` formula, which is inverted relative to the standard S2
convention (see the comment at its computation below).
"""

import math

import numpy as np
import pandas as pd

from ..data.tables import get_rci_tables
from ..utils.constants import _AA_3TO1

# RCI atom name -> makeshift Atom_ID
_ATOM_TO_MAKESHIFT = {"N": "N", "CO": "C", "CA": "CA", "CB": "CB", "NH": "H", "HA": "HA"}

# Atom types actually used in the sigma calculation (H/NH excluded by default),
# in the exact order the reference script processes them (write_atom_list()).
_CALC_ATOMS = ["CA", "CB", "CO", "N", "HA"]

_HERTZ = {"CA": 2.5, "CB": 2.5, "CO": 2.5, "N": 1.0, "HA": 10.0}
_COEF = {"CA": 0.72, "CB": 0.15, "CO": 0.72, "N": 0.59, "HA": 0.85}

_EARLY_FLOOR_POS = 0.1
_EARLY_FLOOR_NEG = -0.1
_FLOOR1 = 0.5      # per-atom Hertz-scaled deviation floor
_FLOOR = 0.5        # sigma ceiling before termini correction
_FLOOR2 = 0.6        # sigma ceiling after termini correction
_GAP_LIMIT = 2
_SCALE = 1.125
_OXIDIZED_CYS_CB_PPM = 35.0


def _build_simpred(seq_map, tables):
    """Reference ('random coil + neighbor correction') shift per residue.

    Returns a DataFrame indexed by residue number with columns
    [N, C, CA, CB, H, HA] (makeshift atom names); NaN where undefined
    (e.g. Gly CB, Pro N/H).
    """
    rc = tables["random_coil"]
    preceed = tables["preceed_effect"]
    nxt = tables["next_effect"]
    preceed2 = tables["preceed_preceed_effect"]
    next2 = tables["next_next_effect"]
    cols = list(rc.columns)  # [N, CO, CA, CB, NH, HA]

    def lookup(table, aa):
        if aa is None or aa not in table.index:
            return pd.Series(0.0, index=cols)
        return table.loc[aa]

    rows = {}
    for resnum, aa_i in seq_map.items():
        coil = rc.loc[aa_i] if aa_i in rc.index else pd.Series(np.nan, index=cols)
        total = (
            coil
            + lookup(nxt, seq_map.get(resnum + 1))
            + lookup(preceed, seq_map.get(resnum - 1))
            + lookup(next2, seq_map.get(resnum + 2))
            + lookup(preceed2, seq_map.get(resnum - 2))
        )
        rows[resnum] = total

    simpred = pd.DataFrame.from_dict(rows, orient="index")[cols]
    simpred = simpred.rename(columns=_ATOM_TO_MAKESHIFT)
    simpred.index.name = "Seq_ID"
    return simpred.sort_index()


def _early_floor_clamp(diff, atom):
    hz = _HERTZ[atom]
    pos_floor = _EARLY_FLOOR_POS / hz
    neg_floor = _EARLY_FLOOR_NEG / hz
    if 0 < diff < pos_floor:
        return pos_floor
    if neg_floor < diff < 0:
        return neg_floor
    return diff


def _raw_deviation(observed_multi, simpred, atom):
    """{residue: [abs deviation, ...]} for one atom type, with the early-floor
    clamp. `observed_multi` maps residue -> list of observed shift values
    (almost always singleton; see note on Gly HA2/HA3 below)."""
    out = {}
    for resnum, obs_vals in observed_multi.items():
        simpred_val = simpred.get(resnum)
        if simpred_val is None or (isinstance(simpred_val, float) and np.isnan(simpred_val)):
            continue
        devs = []
        for obs_val in obs_vals:
            diff = obs_val - simpred_val
            diff = _early_floor_clamp(diff, atom)
            devs.append(abs(diff))
        out[resnum] = devs
    return out


def _gap_fill(raw, all_residues, bound_first, bound_last, gap_limit=_GAP_LIMIT):
    """gap_fill2(): fill missing (residue, atom) abs-deviations from the
    nearest observed value up to `gap_limit` residues on each side
    (independently). `raw` maps residue -> list of values (see
    _raw_deviation): a direct match is passed through unaveraged (a Gly
    residue keeps both its HA2/HA3 entries), but a *borrowed* neighbor's
    value(s) are immediately averaged to a single number, matching
    gap_fill2()'s pos_neg_list_*_ave."""
    filled = {}
    for r in all_residues:
        if r in raw:
            filled[r] = list(raw[r])
            continue
        if not (bound_first <= r <= bound_last):
            continue
        found = []
        for i in range(1, gap_limit + 1):
            if (r + i) in raw:
                found.extend(raw[r + i])
                break
        for i in range(1, gap_limit + 1):
            if (r - i) in raw:
                found.extend(raw[r - i])
                break
        if found:
            filled[r] = [sum(found) / len(found)]
    return filled


def _smooth3(values, gap_limit=_GAP_LIMIT):
    """smoothing()/final_smoothing() specialized to smooth_factor=3
    (the script's effective default despite its misleading top-of-file
    flags -- see rci_v_1c.py lines 1138-1163).

    `values`: {residue: [float, ...]}. Almost always a singleton list;
    a residue can carry >1 raw value because the reference script matches
    observed atoms by a 2-character name prefix, which for atom type "HA"
    also matches Gly's HA2/HA3 (both slice to "HA") -- so a Gly residue
    contributes two values instead of one, and any 3-window touching it
    totals >3 values (see rci_v_1c.py lines 2159-2183, the branch that
    handles "more than smooth_factor" by just averaging everything found,
    with no borrowing).

    Returns {residue: float} covering [min(values), max(values)].
    """
    if not values:
        return {}
    residues = sorted(values)
    first_residue = residues[0]
    last_residue = residues[-1]
    result = {}

    # N-terminus: only the very first residue, forward-only 1-neighbor average.
    r0 = first_residue
    collected = list(values[r0])
    for d in (r0 + 1, r0 + 2):
        if d in values:
            collected.extend(values[d])
            break
    result[r0] = sum(collected) / len(collected)

    # C-terminus: only the very last residue, backward-only 1-neighbor average.
    rN = last_residue
    collected = list(values[rN])
    for d in (rN - 1, rN - 2):
        if d in values:
            collected.extend(values[d])
            break
    result[rN] = sum(collected) / len(collected)

    # Main sliding 3-window; center ranges over (first_residue, last_residue) exclusive.
    for r in range(first_residue, last_residue - 1):
        center = r + 1
        window = (r, r + 1, r + 2)
        slots = [values.get(w) for w in window]
        total_count = sum(len(v) for v in slots if v is not None)
        collected = []
        for v in slots:
            if v is not None:
                collected.extend(v)

        if total_count == 3:
            result[center] = sum(collected) / 3
            continue

        if total_count > 3:
            # No borrowing needed -- just average whatever the window has.
            result[center] = sum(collected) / len(collected)
            continue

        missing = [w for w, v in zip(window, slots) if v is None]
        bigger_missing = sum(1 for m in missing if m >= center)
        smaller_missing = sum(1 for m in missing if m < center)
        more_smaller = 0
        more_bigger = 0

        if bigger_missing > 0:
            found = 0
            new_end = r + 2
            while found != bigger_missing:
                new_end += 1
                if new_end < last_residue:
                    if new_end < (r + 3 + gap_limit):
                        if new_end in values:
                            collected.extend(values[new_end])
                            found += len(values[new_end])
                    else:
                        break
                else:
                    more_smaller = bigger_missing - found
                    break

        if smaller_missing > 0:
            found = 0
            new_start = r
            while found != smaller_missing:
                new_start -= 1
                if new_start >= first_residue:
                    if new_start >= (r - gap_limit):
                        if new_start in values:
                            collected.extend(values[new_start])
                            found += len(values[new_start])
                    else:
                        break
                else:
                    more_bigger = smaller_missing - found
                    break

        if more_bigger > 0:
            found = 0
            new_end = r + 2
            while found != more_bigger:
                new_end += 1
                if new_end <= last_residue and new_end < (r + 2 + gap_limit):
                    if new_end in values:
                        collected.extend(values[new_end])
                        found += len(values[new_end])
                else:
                    break

        if more_smaller > 0:
            found = 0
            new_start = r
            while found != more_smaller:
                new_start -= 1
                if new_start >= first_residue and new_start >= (r - gap_limit):
                    if new_start in values:
                        collected.extend(values[new_start])
                        found += len(values[new_start])
                else:
                    break

        if collected:
            result[center] = sum(collected) / len(collected)

    return result


def _combine_sigma(smoothed_by_atom, all_residues):
    """function_flag==8: combine per-atom smoothed abs deviations into a
    per-residue sigma."""
    sigma = {}
    for r in all_residues:
        contributions = []
        for atom in _CALC_ATOMS:
            v = smoothed_by_atom[atom].get(r)
            if v is None:
                continue
            v = v * _HERTZ[atom]
            if abs(v) < _FLOOR1:
                v = _FLOOR1 if v >= 0 else -_FLOOR1
            contributions.append(v * _COEF[atom] * 5)
        if not contributions:
            continue
        mean_v = sum(contributions) / len(contributions)
        value_abs = 0.0
        if mean_v != 0:
            value_abs = 1.0 / (abs(mean_v) ** 1.5)
        sigma[r] = min(value_abs, _FLOOR)
    return sigma


def _end_effect3(sigma, first_residue, last_residue):
    """Termini correction (termini_corr_flag==3): reflect sigma up toward
    the local terminal max within 3 residues of each chain end."""
    n_end = [(sigma[r], r) for r in sigma if abs(r - first_residue) <= 4]
    c_end = [(sigma[r], r) for r in sigma if abs(last_residue - r) <= 4]

    n_max = n_max_place = None
    if n_end:
        n_max, n_max_place = max(n_end, key=lambda x: x[0])
    c_max = c_max_place = None
    if c_end:
        c_max, c_max_place = max(c_end, key=lambda x: x[0])

    result = dict(sigma)
    for r, s in sigma.items():
        if abs(r - first_residue) <= 3:
            if n_max is not None and s < n_max and r < n_max_place:
                s2 = s + 2 * abs(s - n_max)
                result[r] = min(s2, _FLOOR2)
        elif abs(last_residue - r) <= 3:
            if c_max is not None and s < c_max and r > c_max_place:
                s2 = s + 2 * abs(s - c_max)
                result[r] = min(s2, _FLOOR2)
    return result


class RCI:
    """
    Predict per-residue backbone flexibility (Random Coil Index) from
    assigned NMR chemical shifts. Pure Python/pandas -- no external binary.

        r = RCI.from_bmrb(4403)
        r.run()
        r.results   # Seq_ID, Comp_ID, RCI, S2, MD_RMSD, NMR_RMSD

    Or, starting from a :class:`~makeshift.chemshift.ChemicalShifts` you
    already have (e.g. from ``ChemicalShifts.from_bmrb(...)``), compute in
    one step with :meth:`calc`:

        cs = ChemicalShifts.from_bmrb(4403, keep_download=True)
        r = RCI.calc(cs)
        r.results
    """

    def __init__(self, shifts=None, sequence=None, first_resid=None, entry_id=None,
                 entity_id=None, entry=None):
        self.shifts = shifts
        self.sequence = sequence
        self.first_resid = first_resid
        self.entry_id = entry_id
        self.entity_id = entity_id
        self.entry = entry
        self.results = None

    @staticmethod
    def _resolve_sequence(entry, entity_id=None):
        seqs = entry.sequences()
        if entity_id is not None:
            sequence = entry.sequences(entity_id=entity_id)
        else:
            poly = seqs[
                seqs["Polymer_type"].str.contains("polypeptide", case=False, na=False)
            ]
            if poly.empty:
                raise ValueError(
                    f"No polypeptide sequence found in entry {getattr(entry, 'entry_id', None)}"
                )
            sequence = poly.iloc[0]["Polymer_seq_one_letter_code"]
            entity_id = poly.iloc[0]["ID"]
        if not isinstance(sequence, str) or not sequence or pd.isna(sequence):
            raise ValueError("could not resolve a sequence; pass sequence=... explicitly")
        return sequence, entity_id

    @classmethod
    def from_bmrb(cls, bmrb_id, entity_id=None, sequence=None, **fetch_kw):
        from ..entry import NMRStarEntry
        entry = NMRStarEntry.from_bmrb(bmrb_id, **fetch_kw)
        return cls.from_entry(entry, entity_id=entity_id, sequence=sequence)

    @classmethod
    def from_entry(cls, entry, entity_id=None, sequence=None):
        from ..chemshift import ChemicalShifts

        if sequence is None:
            sequence, entity_id = cls._resolve_sequence(entry, entity_id)

        shifts = ChemicalShifts.from_entry(entry).data
        if shifts.empty:
            raise ValueError(
                f"No backbone chemical shifts in entry {getattr(entry, 'entry_id', None)}"
            )
        first_resid = int(shifts["Seq_ID"].min())
        return cls(shifts, sequence, first_resid=first_resid,
                    entry_id=getattr(entry, "entry_id", None),
                    entity_id=entity_id, entry=entry)

    @classmethod
    def calc(cls, chemshifts, entity_id=None, sequence=None):
        """
        Compute RCI directly from a :class:`ChemicalShifts` object (e.g.
        ``ChemicalShifts.from_bmrb(...)``) and return the populated,
        already-run :class:`RCI`. Resolves the polymer sequence from the
        ChemicalShifts' underlying entry unless `sequence` is given.
        """
        entry = chemshifts.entry
        if sequence is None:
            if entry is None:
                raise ValueError(
                    "chemshifts has no associated entry; pass sequence=... explicitly"
                )
            sequence, entity_id = cls._resolve_sequence(entry, entity_id)

        shifts = chemshifts.data
        if shifts.empty:
            raise ValueError("chemshifts has no chemical shift data")
        first_resid = int(shifts["Seq_ID"].min())
        obj = cls(shifts, sequence, first_resid=first_resid,
                  entry_id=getattr(entry, "entry_id", None),
                  entity_id=entity_id, entry=entry)
        obj.run()
        return obj

    def _seq_map(self):
        if self.sequence is not None:
            first_resid = self.first_resid if self.first_resid is not None else 1
            return {
                first_resid + i: aa.upper()
                for i, aa in enumerate(self.sequence)
            }
        import warnings
        warnings.warn(
            "No sequence supplied; inferring from the chemical shift table "
            "(neighbor-residue corrections will be missing for any residue "
            "without an observed shift).",
            stacklevel=3,
        )
        seq = self.shifts[["Seq_ID", "Comp_ID"]].drop_duplicates().sort_values("Seq_ID")
        return {
            int(row.Seq_ID): _AA_3TO1.get(row.Comp_ID.upper(), "X")
            for row in seq.itertuples()
        }

    def run(self):
        seq_map = self._seq_map()
        first_residue = min(seq_map)
        last_residue = max(seq_map)
        all_residues = list(range(first_residue, last_residue + 1))

        tables = get_rci_tables()
        simpred = _build_simpred(seq_map, tables)

        oxidized_cys = set(
            self.shifts.loc[
                (self.shifts["Comp_ID"].str.upper() == "CYS")
                & (self.shifts["Atom_ID"] == "CB")
                & (self.shifts["Val"] > _OXIDIZED_CYS_CB_PPM),
                "Seq_ID",
            ].astype(int)
        )

        cum_first, cum_last = None, None
        smoothed_by_atom = {}
        for atom in _CALC_ATOMS:
            makeshift_atom = _ATOM_TO_MAKESHIFT[atom]
            # Match atom names by 2-char prefix, exactly like the reference
            # script's `atom_name[0:2]==atom_type` -- this deliberately also
            # matches Gly's HA2/HA3 when atom=="HA" (see _smooth3 docstring).
            obs = self.shifts[self.shifts["Atom_ID"].str[:2] == makeshift_atom]
            if atom == "CB":
                obs = obs[~obs["Seq_ID"].isin(oxidized_cys)]
            observed_multi = {}
            for r in obs.itertuples():
                if pd.notna(r.Val):
                    observed_multi.setdefault(int(r.Seq_ID), []).append(float(r.Val))

            simpred_col = simpred[makeshift_atom]
            raw = _raw_deviation(observed_multi, simpred_col, atom)

            if raw:
                atom_first, atom_last = min(raw), max(raw)
                cum_first = atom_first if cum_first is None else min(cum_first, atom_first)
                cum_last = atom_last if cum_last is None else max(cum_last, atom_last)

            bound_first = cum_first if cum_first is not None else first_residue
            bound_last = cum_last if cum_last is not None else last_residue
            filled = _gap_fill(raw, all_residues, bound_first, bound_last)
            smoothed_by_atom[atom] = _smooth3(filled)

        sigma = _combine_sigma(smoothed_by_atom, all_residues)
        sigma = _end_effect3(sigma, first_residue, last_residue)
        sigma = _smooth3({r: [v] for r, v in sigma.items()})

        rows = []
        for r in sorted(sigma):
            s = sigma[r]
            rci = s / _SCALE
            rows.append({
                "Seq_ID": r,
                "Comp_ID": seq_map.get(r),
                "RCI": rci,
                # Published relation (Berjanskii & Wishart 2005): canonical
                # S2 convention, rigid -> ~1, flexible -> lower. Note this
                # differs from rci_v_1c.py's own .S2.txt output (0.4*ln(1+
                # 17.7*RCI)), which increases with RCI -- the opposite of
                # the standard S2 sense -- and is not reproduced here.
                "S2": 1 - 0.5 * math.log(1 + 10 * rci),
                "MD_RMSD": rci * 29.55,
                "NMR_RMSD": rci * 16.44,
            })

        self.results = pd.DataFrame(rows)
        return self

    def __repr__(self):
        n = self.shifts["Seq_ID"].nunique() if isinstance(self.shifts, pd.DataFrame) else "?"
        state = "run" if self.results is not None else "not run"
        return f"<RCI entry={self.entry_id} residues={n} ({state})>"
