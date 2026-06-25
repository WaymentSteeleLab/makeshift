"""
Backbone amide 1H/15N assignments as an object.
"""
import pandas as pd
import warnings

from .entry import NMRStarEntry
from .chemshift import ChemicalShifts

from .utils.constants import _AA_3TO1, _AA_1TO3

_OUT_COLS = ["Seq_ID", "Auth_seq_ID", "Comp_ID", "H_ppm", "N_ppm", "assn_label"]


class PeakList:

    def __init__(self, data, source=None):
        self.data = data
        self.source = source
        self.entry = None
        self.entity_id = None
        self.cs_saveframe = None

    @classmethod
    def from_chemshifts(cls, cs, cs_saveframe=None, entity_id=None):
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
        atoms = set(sub["Atom_ID"].unique())
        has_n = "N" in atoms
        has_h = bool({"H", "HN"} & atoms)
        
        if not (has_n and has_h):
            missing = [a for a, present in (("N", has_n), ("H/HN", has_h)) if not present]
            warnings.warn(
                f"{source}: chemical shift saveframe {chosen!r} has no {' or '.join(missing)} "
                f"shifts — can't build a backbone amide peak list. Returning an empty PeakList.",
                UserWarning,
            )
            obj = cls(pd.DataFrame(columns=_OUT_COLS), source=source)
            obj.entry = cs.entry
            obj.entity_id = chosen_entity
            obj.cs_saveframe = chosen
            return obj
 
        n_df = (sub[sub["Atom_ID"] == "N"]
                [["Seq_ID", "Auth_seq_ID", "Comp_ID", "Val"]]
                .rename(columns={"Val": "N_ppm"}))
        h_df = (sub[sub["Atom_ID"].isin(["H", "HN"])]
                [["Seq_ID", "Val"]]
                .rename(columns={"Val": "H_ppm"}))
 
        out = (n_df.merge(h_df, on="Seq_ID")
                   .dropna(subset=["H_ppm", "N_ppm"])
                   .reset_index(drop=True))
        if out.empty:
            warnings.warn(
                f"{source}: saveframe {chosen!r} has N and H/HN shifts, but none "
                f"share a residue — no backbone amide pairs. Returning an empty PeakList.",
                UserWarning,
            )
        out = cls._label(out)
 
        obj = cls(out[_OUT_COLS], source=source)
        obj.entry = cs.entry
        obj.entity_id = chosen_entity
        obj.cs_saveframe = chosen
        return obj

    @classmethod
    def from_entry(cls, entry, cs_saveframe=None):
        cs = ChemicalShifts.from_entry(entry)
        return cls.from_chemshifts(cs, cs_saveframe=cs_saveframe)

    @classmethod
    def from_bmrb(cls, bmrb_id, cs_saveframe=None, **fetch_kw):
        entry = NMRStarEntry.from_bmrb(bmrb_id, **fetch_kw)
        return cls.from_entry(entry, cs_saveframe=cs_saveframe)

    # local CSV source  (res / shift / atom)

    @classmethod
    def from_csv(cls, path, seq_offset=0):
        df = pd.read_csv(path).copy()
        df["aa_1"] = df["res"].str.extract(r"^([A-Za-z])")
        df["Auth_seq_ID"] = df["res"].str.extract(r"(\d+)").astype(int)
        df["Comp_ID"] = df["aa_1"].map(_AA_1TO3)

        n_df = (df[df["atom"] == "15N"]
                [["Auth_seq_ID", "Comp_ID", "shift"]]
                .rename(columns={"shift": "N_ppm"}))
        h_df = (df[df["atom"] == "1H"]
                [["Auth_seq_ID", "shift"]]
                .rename(columns={"shift": "H_ppm"}))

        out = (n_df.merge(h_df, on="Auth_seq_ID")
                   .dropna(subset=["H_ppm", "N_ppm"])
                   .reset_index(drop=True))
        out["Seq_ID"] = out["Auth_seq_ID"] + seq_offset
        out = cls._label(out)
        print(f"  {len(out)} backbone amide assignments from {path}")
        return cls(out[_OUT_COLS], source=str(path))

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