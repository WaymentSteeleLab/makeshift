"""
Random Coil Index: per-residue backbone flexibility from assigned shifts.

Port of the reference script ``rci_v_1c.py`` (Berjanskii & Wishart 2005),
following the code path that script takes under its own defaults —
Schwarzinger random coil and neighbor corrections, 3-point smoothing,
``end_effect3`` termini correction, ``function_flag==8`` sigma combination.
The alternate branches behind its other CLI flags are not reproduced.

S2 is the one deliberate departure: the published Berjanskii & Wishart
relation is used rather than the script's own ``.S2.txt`` formula, which runs
the opposite way. The TALOS-N backend lives in :mod:`makeshift.rci._talosn`;
both are validated in ``docs/rci_validation.md``.
"""

import math
import warnings

import numpy as np
import pandas as pd

from ..data.tables import get_rci_tables
from ..utils.constants import _AA_3TO1
from ._talosn import run_talosn_rci

_ALGORITHMS = ("wishart", "talosn")

# RCI atom name -> makeshift Atom_ID
_ATOM_TO_MAKESHIFT = {"N": "N", "CO": "C", "CA": "CA", "CB": "CB", "NH": "H", "HA": "HA"}

# Atoms used in the sigma sum (NH excluded), in the order the reference script
# processes them — the order matters, see the gap-fill bounds in _run_wishart.
_CALC_ATOMS = ["CA", "CB", "CO", "N", "HA"]

_HERTZ = {"CA": 2.5, "CB": 2.5, "CO": 2.5, "N": 1.0, "HA": 10.0}
_COEF = {"CA": 0.72, "CB": 0.15, "CO": 0.72, "N": 0.59, "HA": 0.85}

_EARLY_FLOOR = 0.1          # min |deviation| in Hz, before averaging
_DEV_FLOOR = 0.5            # min |deviation| in Hz, per atom in the sigma sum
_SIGMA_MAX = 0.5            # sigma ceiling, before the termini correction
_SIGMA_MAX_TERMINI = 0.6    # sigma ceiling, after it
_GAP_LIMIT = 2
_SCALE = 1.125
_OXIDIZED_CYS_CB_PPM = 35.0


def _build_simpred(seq_map, tables):
    """
    Reference shift per residue: random coil plus +-1 and +-2 neighbor
    corrections. Returns a DataFrame indexed by residue number, columns
    [N, C, CA, CB, H, HA], NaN where undefined (Gly CB, Pro N/H).
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
    """Push a nonzero deviation out to at least +-0.1 Hz so it survives the mean."""
    floor = _EARLY_FLOOR / _HERTZ[atom]
    if 0 < diff < floor:
        return floor
    if -floor < diff < 0:
        return -floor
    return diff


def _raw_deviation(observed_multi, simpred, atom):
    """
    {residue: [abs deviation, ...]} for one atom type, clamped by
    _early_floor_clamp. `observed_multi` maps residue -> list of observed
    values; it is a list only because Gly contributes both HA2 and HA3.
    """
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
    """
    gap_fill2(): fill a missing (residue, atom) deviation from the nearest
    observed value up to `gap_limit` residues away on each side.

    A residue with its own value passes through untouched, so Gly keeps both
    HA2 and HA3. Borrowed values are averaged down to one number first.
    """
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
    """
    smoothing()/final_smoothing() at smooth_factor=3, the script's effective
    default. Returns {residue: float} over [min(values), max(values)].

    `values` maps residue -> list of floats. The list is a singleton except
    at Gly, which contributes both HA2 and HA3 (the 2-character atom-name
    match treats them as one atom type), so a window touching a Gly holds
    more than 3 values. In that case the script averages everything it has
    and skips the borrowing below.
    """
    if not values:
        return {}
    residues = sorted(values)
    first_residue = residues[0]
    last_residue = residues[-1]
    result = {}

    # N-terminus: first residue only, averaged forward with one neighbor.
    r0 = first_residue
    collected = list(values[r0])
    for d in (r0 + 1, r0 + 2):
        if d in values:
            collected.extend(values[d])
            break
    result[r0] = sum(collected) / len(collected)

    # C-terminus: last residue only, averaged backward with one neighbor.
    rN = last_residue
    collected = list(values[rN])
    for d in (rN - 1, rN - 2):
        if d in values:
            collected.extend(values[d])
            break
    result[rN] = sum(collected) / len(collected)

    # Sliding 3-window over every interior residue.
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
            # A Gly is in the window; average what's there, don't borrow.
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
    """
    function_flag==8: collapse the per-atom smoothed deviations into one
    sigma per residue. Deviations go to Hz, get floored, are weighted by
    atom, and the mean is inverted — small deviations mean a rigid residue.
    """
    sigma = {}
    for r in all_residues:
        contributions = []
        for atom in _CALC_ATOMS:
            v = smoothed_by_atom[atom].get(r)
            if v is None:
                continue
            v = v * _HERTZ[atom]
            if abs(v) < _DEV_FLOOR:
                v = _DEV_FLOOR if v >= 0 else -_DEV_FLOOR
            contributions.append(v * _COEF[atom] * 5)
        if not contributions:
            continue
        mean_v = sum(contributions) / len(contributions)
        value_abs = 0.0
        if mean_v != 0:
            value_abs = 1.0 / (abs(mean_v) ** 1.5)
        sigma[r] = min(value_abs, _SIGMA_MAX)
    return sigma


def _end_effect3(sigma, first_residue, last_residue):
    """
    end_effect3(): within 3 residues of either terminus, pull sigma up
    toward the local maximum. Termini are flexible and the smoothing above
    otherwise drags them down toward the ordered core.
    """
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
                result[r] = min(s2, _SIGMA_MAX_TERMINI)
        elif abs(last_residue - r) <= 3:
            if c_max is not None and s < c_max and r > c_max_place:
                s2 = s + 2 * abs(s - c_max)
                result[r] = min(s2, _SIGMA_MAX_TERMINI)
    return result


class RCI:
    """
    Per-residue backbone flexibility from assigned chemical shifts. Pure
    Python — no external binary, unlike :class:`~makeshift.talosn.TalosN`.

        r = RCI.from_bmrb(4403)
        r.run()
        r.results     # Seq_ID, Comp_ID, RCI, S2

    Or, from a :class:`~makeshift.chemshift.ChemicalShifts` you already have,
    in one step:

        cs = ChemicalShifts.from_bmrb(4403, keep_download=True)
        RCI.calc(cs).results

    Parameters
    ----------
    algorithm : {'wishart', 'talosn'}
        Which calculation to run. ``'wishart'`` ports the reference script
        ``rci_v_1c.py``; ``'talosn'`` ports the separate RCI-S2 module
        bundled inside TALOS-N, which reproduces its quirks so the output
        matches the compiled binary (see :mod:`makeshift.rci._talosn`).
        The two agree closely on RCI but report S2 on different scales —
        see ``docs/rci_validation.md``.
    neighbor_table : {'schwarzinger', 'wang', 'schwartz_wang'}
        Which preceding/next-residue corrections to use; see
        :data:`makeshift.data.tables.RCI_NEIGHBOR_TABLES`. Nothing here
        predicts secondary structure, so all three resolve to their
        coil-state values. **Ignored when algorithm='talosn'**, whose
        tables are compiled into the binary.
    """

    def __init__(self, shifts=None, sequence=None, first_resid=None, entry_id=None,
                 entity_id=None, entry=None, neighbor_table="schwarzinger",
                 algorithm="wishart"):
        if algorithm not in _ALGORITHMS:
            raise ValueError(f"algorithm must be one of {_ALGORITHMS}, got {algorithm!r}")
        self.shifts = shifts
        self.sequence = sequence
        self.first_resid = first_resid
        self.entry_id = entry_id
        self.entity_id = entity_id
        self.entry = entry
        self.neighbor_table = neighbor_table
        self.algorithm = algorithm
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
    def from_bmrb(cls, bmrb_id, entity_id=None, sequence=None,
                  neighbor_table="schwarzinger", algorithm="wishart", **fetch_kw):
        from ..entry import NMRStarEntry
        entry = NMRStarEntry.from_bmrb(bmrb_id, **fetch_kw)
        return cls.from_entry(entry, entity_id=entity_id, sequence=sequence,
                              neighbor_table=neighbor_table, algorithm=algorithm)

    @classmethod
    def from_entry(cls, entry, entity_id=None, sequence=None,
                    neighbor_table="schwarzinger", algorithm="wishart"):
        from ..chemshift import ChemicalShifts

        if sequence is None:
            sequence, entity_id = cls._resolve_sequence(entry, entity_id)

        shifts = ChemicalShifts.from_entry(entry).data
        if shifts.empty:
            raise ValueError(
                f"No backbone chemical shifts in entry {getattr(entry, 'entry_id', None)}"
            )
        first_resid = entry.resolve_first_resid(entity_id, sequence, shifts)
        return cls(shifts, sequence, first_resid=first_resid,
                    entry_id=getattr(entry, "entry_id", None),
                    entity_id=entity_id, entry=entry, neighbor_table=neighbor_table,
                    algorithm=algorithm)

    @classmethod
    def calc(cls, chemshifts, entity_id=None, sequence=None,
             neighbor_table="schwarzinger", algorithm="wishart"):
        """
        Build from a :class:`ChemicalShifts`, run, and return the populated
        :class:`RCI`. The sequence comes from the ChemicalShifts' entry
        unless `sequence` is passed.
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
        if entry is not None:
            first_resid = entry.resolve_first_resid(entity_id, sequence, shifts)
        else:
            first_resid = int(shifts["Seq_ID"].min())
        obj = cls(shifts, sequence, first_resid=first_resid,
                  entry_id=getattr(entry, "entry_id", None),
                  entity_id=entity_id, entry=entry, neighbor_table=neighbor_table,
                  algorithm=algorithm)
        obj.run()
        return obj

    def _seq_map(self):
        """{residue number: one-letter code} over the whole polymer."""
        if self.sequence is not None:
            first_resid = self.first_resid if self.first_resid is not None else 1
            return {
                first_resid + i: aa.upper()
                for i, aa in enumerate(self.sequence)
            }
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
        """Run the selected algorithm and populate :attr:`results`."""
        if self.algorithm == "talosn":
            self._run_talosn()
        else:
            self._run_wishart()
        return self

    def _run_talosn(self):
        seq_map = self._seq_map()
        tables = get_rci_tables(neighbor_table="schwarzinger")
        simpred = _build_simpred(seq_map, tables)
        self.results = run_talosn_rci(self.shifts, seq_map, simpred)
        return self

    def _run_wishart(self):
        seq_map = self._seq_map()
        first_residue = min(seq_map)
        last_residue = max(seq_map)
        all_residues = list(range(first_residue, last_residue + 1))

        tables = get_rci_tables(neighbor_table=self.neighbor_table)
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
            # 2-character prefix match, as in the reference script. This is
            # what picks up Gly's HA2/HA3 under atom type "HA".
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
                # Berjanskii & Wishart 2005 as published: rigid -> ~1.
                # rci_v_1c.py's own .S2.txt writes 0.4*ln(1+17.7*RCI),
                # which climbs with RCI; not reproduced here.
                "S2": 1 - 0.5 * math.log(1 + 10 * rci),
            })

        self.results = pd.DataFrame(rows)
        return self

    def __repr__(self):
        n = self.shifts["Seq_ID"].nunique() if isinstance(self.shifts, pd.DataFrame) else "?"
        state = "run" if self.results is not None else "not run"
        return f"<RCI entry={self.entry_id} residues={n} ({state})>"
