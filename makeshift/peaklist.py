"""
Per-residue peak assignments as an object.

By default this is the backbone amide (H, N) pair used for 2D HSQC-type
peak lists, but ``dims`` lets you build peak lists over any set of atoms
"""
import re
import pandas as pd
import warnings

from .entry import NMRStarEntry
from .chemshift import ChemicalShifts

from .utils.constants import _AA_3TO1, _AA_1TO3

_DEFAULT_DIMS = ("H", "N")
_CANONICAL_LABEL = {"HN": "H"}
_ATOM_ALIASES = {"H": ("H", "HN"), "HN": ("H", "HN")}
_CSV_DEFAULT_DIMS = (("H", ("1H",)), ("N", ("15N",)))


class PeakList:

    def __init__(self, data, source=None):
        self.data = data
        self.source = source
        self.entry = None
        self.entity_id = None
        self.cs_saveframe = None

    @staticmethod
    def _normalize_dims(dims):
        """
        Accept either a flat sequence of atom labels (``("H", "N")``) or the
        fuller ``(label, acceptable_atom_names)`` form for custom aliasing
        """
        out = []
        for d in dims:
            if isinstance(d, str):
                label = _CANONICAL_LABEL.get(d, d)
                out.append((label, _ATOM_ALIASES.get(d, (d,))))
            else:
                label, names = d
                names = (names,) if isinstance(names, str) else tuple(names)
                out.append((label, names))
        return tuple(out)

    @staticmethod
    def _out_cols(dims):
        return (["Seq_ID", "Auth_seq_ID", "Comp_ID"]
                + [f"{label}_ppm" for label, _ in dims]
                + ["assn_label"])

    @staticmethod
    def _merge_dims(sub, dims, id_col="Seq_ID", atom_col="Atom_ID", val_col="Val",
                     base_cols=("Auth_seq_ID", "Comp_ID")):
        atoms_present = set(sub[atom_col].unique())
        missing = [label for label, names in dims if not (set(names) & atoms_present)]
        if missing:
            return None, missing

        first_label, first_names = dims[0]
        out = (sub[sub[atom_col].isin(first_names)]
               [[id_col, *base_cols, val_col]]
               .rename(columns={val_col: f"{first_label}_ppm"}))

        for label, names in dims[1:]:
            dim_df = (sub[sub[atom_col].isin(names)]
                      [[id_col, val_col]]
                      .rename(columns={val_col: f"{label}_ppm"}))
            out = out.merge(dim_df, on=id_col)

        ppm_cols = [f"{label}_ppm" for label, _ in dims]
        out = out.dropna(subset=ppm_cols).reset_index(drop=True)
        return out, []

    @classmethod
    def from_chemshifts(cls, cs, cs_saveframe=None, entity_id=None, dims=None):
        dims = cls._normalize_dims(dims or _DEFAULT_DIMS)
        df = cs.data
 
        ids = df["ChemShift_ID"].unique()
        chosen = cs_saveframe or ids[0]
        if cs_saveframe is None and len(ids) > 1:
            print(f"  Note: {len(ids)} chemical shift saveframes — using first "
                  f"({chosen}). Others: {list(ids[1:])}")
 
        if cs_saveframe is not None:
            df = df[df["ChemShift_ID"] == cs_saveframe]
            if df.empty:
                raise ValueError(f"no chemical shifts for ChemShift_ID={cs_saveframe} in chemical shifts present: {list(ids)}")

        entities = df["Entity_ID"].dropna().unique()
        if entity_id is not None:
            chosen_entity = int(entity_id)
            sub = df[df["Entity_ID"] == chosen_entity]
            if sub.empty:
                raise ValueError(f"no chemical shifts for Entity_ID={entity_id!r} "
                                 f"in saveframe {chosen!r}; entities present: {list(entities)}")
        elif len(entities) == 0:
            raise ValueError("No entities in entry.")
        else:
            chosen_entity = int(entities[0])
            if len(entities) > 1:
                print(f"  Note: {len(entities)} entities in this saveframe — using "
                      f"first (Entity_ID={chosen_entity}). Others: {list(entities[1:])}")
            sub = df[df["Entity_ID"] == chosen_entity]
        
        eid = cs.entry.entry_id if cs.entry is not None else None
        source = f"bmrb:{eid}"
        out_cols = cls._out_cols(dims)
        labels = [label for label, _ in dims]

        out, missing = cls._merge_dims(sub, dims)
        if missing:
            warnings.warn(
                f"{source}: chemical shift saveframe {chosen!r} has no {' or '.join(missing)} "
                f"shifts — can't build a peak list for dims {labels}. Returning an empty PeakList.",
                UserWarning,
            )
            obj = cls(pd.DataFrame(columns=out_cols), source=source)
            obj.entry = cs.entry
            obj.entity_id = chosen_entity
            obj.cs_saveframe = chosen
            return obj

        if out.empty:
            warnings.warn(
                f"{source}: saveframe {chosen!r} has {'/'.join(labels)} shifts, but none "
                f"share a residue — no peaks with all dims present. Returning an empty PeakList.",
                UserWarning,
            )
        out = cls._label(out)

        obj = cls(out[out_cols], source=source)
        obj.entry = cs.entry
        obj.entity_id = chosen_entity
        obj.cs_saveframe = chosen
        return obj

    @classmethod
    def from_entry(cls, entry, cs_saveframe=None, entity_id=None, dims=None):
        cs = ChemicalShifts.from_entry(entry)
        return cls.from_chemshifts(cs, cs_saveframe=cs_saveframe, entity_id=entity_id, dims=dims)

    @classmethod
    def from_bmrb(cls, bmrb_id, cs_saveframe=None, entity_id=None, dims=None, **fetch_kw):
        entry = NMRStarEntry.from_bmrb(bmrb_id, **fetch_kw)
        return cls.from_entry(entry, cs_saveframe=cs_saveframe, entity_id=entity_id, dims=dims)

    # local CSV source  (res / shift / atom)

    @classmethod
    def from_csv(cls, path, seq_offset=0, dims=None):
        dims = cls._normalize_dims(dims or _CSV_DEFAULT_DIMS)
        df = pd.read_csv(path).copy()
        df["aa_1"] = df["res"].str.extract(r"^([A-Za-z])")
        df["Auth_seq_ID"] = df["res"].str.extract(r"(\d+)").astype(int)
        df["Comp_ID"] = df["aa_1"].map(_AA_1TO3)

        out_cols = cls._out_cols(dims)
        out, missing = cls._merge_dims(df, dims, id_col="Auth_seq_ID", atom_col="atom",
                                        val_col="shift", base_cols=("Comp_ID",))
        if missing:
            warnings.warn(
                f"{path}: no {' or '.join(missing)} shifts found — can't build a peak "
                f"list for dims {[label for label, _ in dims]}. Returning an empty PeakList.",
                UserWarning,
            )
            return cls(pd.DataFrame(columns=out_cols), source=str(path))

        out["Seq_ID"] = out["Auth_seq_ID"] + seq_offset
        out = cls._label(out)
        if out.empty:
            warnings.warn(
                f"{path}: found {'/'.join(label for label, _ in dims)} shifts, but none "
                "share a residue — no peaks with all dims present.",
                UserWarning,
            )
        print(f"  {len(out)} peaks from {path}")
        return cls(out[out_cols], source=str(path))

    # Sparky peak list  (Assignment  w1  w2 ...)

    _SPARKY_RES_RE = re.compile(r"^([A-Za-z]+?)(-?\d+)([A-Za-z][A-Za-z0-9'\"#*]*)$")

    @classmethod
    def _parse_sparky_assignment(cls, assn):
        """
        Split a Sparky assignment like ``Y4N-H`` or ``Y4N-Y4H`` into
        ``[(res_name, res_num, atom), ...]``, one per dimension. Components
        without a group inherit the previous one (Sparky shorthand).
        Returns None for unassigned / unparseable peaks (e.g. ``?-?``).
        """
        parts = assn.split("-")
        out = []
        prev = None
        for p in parts:
            if not p or "?" in p:
                return None
            m = cls._SPARKY_RES_RE.match(p)
            if m:
                prev = (m.group(1), int(m.group(2)))
                out.append((*prev, m.group(3)))
            elif prev is not None and re.match(r"^[A-Za-z][A-Za-z0-9'\"#*]*$", p):
                out.append((*prev, p))
            else:
                return None
        return out

    @classmethod
    def from_list(cls, path, seq_offset=0, dims=None):
        """
        Read a Sparky peak list (e.g. from ``lt``) with an ``Assignment``
        column followed by one ppm column per dimension (``w1``, ``w2``, ...).
        Each dimension's atom is taken from the matching component of the
        assignment, so ``Y4N-H  118.384  7.508`` gives N=118.384, H=7.508.
        Extra trailing columns (heights, volumes, notes) are ignored, and
        unassigned peaks (``?-?``) are skipped.
        """
        dims = cls._normalize_dims(dims or _DEFAULT_DIMS)
        n_w = None
        rows = []
        n_skipped = 0
        with open(path) as fh:
            for line in fh:
                toks = line.split()
                if not toks:
                    continue
                if toks[0].lower() == "assignment":
                    n_w = sum(1 for t in toks[1:] if re.fullmatch(r"w\d+", t, re.I))
                    continue
                assn = cls._parse_sparky_assignment(toks[0])
                if assn is None:
                    n_skipped += 1
                    continue
                nd = n_w or len(assn)
                try:
                    shifts = [float(t) for t in toks[1:1 + nd]]
                except ValueError:
                    n_skipped += 1
                    continue
                if len(shifts) != nd or len(assn) != nd:
                    n_skipped += 1
                    continue
                for (resname, resnum, atom), shift in zip(assn, shifts):
                    rows.append({"res_name": resname, "Auth_seq_ID": resnum,
                                 "atom": atom, "shift": shift})
        if n_skipped:
            print(f"  skipped {n_skipped} unassigned/unparseable peaks in {path}")

        out_cols = cls._out_cols(dims)
        df = pd.DataFrame(rows, columns=["res_name", "Auth_seq_ID", "atom", "shift"])
        df["Comp_ID"] = [_AA_1TO3.get(r.upper()) if len(r) == 1 else r.upper()
                         for r in df["res_name"]]
        # if a residue/atom appears in several peaks, keep the first
        df = df.drop_duplicates(subset=["Auth_seq_ID", "atom"], keep="first")

        out, missing = cls._merge_dims(df, dims, id_col="Auth_seq_ID", atom_col="atom",
                                        val_col="shift", base_cols=("Comp_ID",))
        if missing:
            warnings.warn(
                f"{path}: no {' or '.join(missing)} shifts found — can't build a peak "
                f"list for dims {[label for label, _ in dims]}. Returning an empty PeakList.",
                UserWarning,
            )
            return cls(pd.DataFrame(columns=out_cols), source=str(path))

        out["Seq_ID"] = out["Auth_seq_ID"] + seq_offset
        out = cls._label(out)
        print(f"  {len(out)} peaks from {path}")
        return cls(out[out_cols], source=str(path))

    @staticmethod
    def _label(out):
        out["assn_label"] = (out["Comp_ID"].map(_AA_3TO1).fillna("?")
                             + out["Seq_ID"].astype(str))
        return out

    def __repr__(self):
        return f"PeakList(residues={len(self.data)}, source={self.source!r})"

    def assignment_string(self, sequence=None, entity_id=None):
        """
        Per-residue label string: 'A' assigned, '.' missing, 'P' proline.

        """
        if sequence is None:
            if self.entry is None:
                raise ValueError("No entry attached; pass an explicit sequence.")
            if entity_id is None:
                entity_id = self.entity_id
            if entity_id is None:
                raise ValueError("No entity_id available; pass entity_id or sequence.")
            sequence = self.entry.sequences(entity_id=entity_id)
        assigned = set(self.data["Seq_ID"])
        chars = []
        for i, aa in enumerate(sequence):
            seqpos = i + 1
            if aa == "P":
                chars.append("P")
            elif seqpos in assigned:
                chars.append("A")
            else:
                chars.append(".")
        return "".join(chars)
