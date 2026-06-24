"""
Backbone amide 1H/15N assignments as an object.
"""
import pandas as pd

from .entry import NMRStarEntry
from .chemshift import ChemicalShifts

from .utils.constants import _AA_3TO1, _AA_1TO3

_OUT_COLS = ["Seq_ID", "Auth_seq_ID", "Comp_ID", "H_ppm", "N_ppm", "assn_label"]


class PeakList:
    """One row per assigned residue: Seq_ID, Auth_seq_ID, Comp_ID, H_ppm, N_ppm."""

    def __init__(self, df, source=None):
        self.df = df
        self.source = source

    # BMRB / NMR-STAR source (via chemical shifts)

    @classmethod
    def from_chemshifts(cls, cs, saveframe=None):
        df = cs.data
 
        ids = df["ChemShift_ID"].unique()
        chosen = saveframe or ids[0]
        if saveframe is None and len(ids) > 1:
            print(f"  Note: {len(ids)} chemical shift saveframes — using first "
                  f"({chosen}). Others: {list(ids[1:])}")
        sub = df[df["ChemShift_ID"] == chosen]
 
        eid = cs.entry.entry_id if cs.entry is not None else None
        source = f"entry:{eid}"
 
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
        return obj

    @classmethod
    def from_entry(cls, entry, saveframe=None):
        cs = ChemicalShifts.from_entry(entry)
        print(cs)
        return cls.from_chemshifts(cs, saveframe=saveframe)

    @classmethod
    def from_bmrb(cls, bmrb_id, saveframe=None, **fetch_kw):
        entry = NMRStarEntry.from_bmrb(bmrb_id, **fetch_kw)
        return cls.from_entry(entry, saveframe=saveframe)

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

    # ------------------------------------------------------------------
    @staticmethod
    def _label(out):
        out["assn_label"] = (out["Comp_ID"].map(_AA_3TO1).fillna("?")
                             + out["Seq_ID"].astype(str))
        return out

    def __repr__(self):
        return f"PeakList(residues={len(self.df)}, source={self.source!r})"