"""
Assigned chemical shifts as an object.
"""
import warnings

import numpy as np
import pandas as pd

from .entry import NMRStarEntry
from .data.tables import get_random_coil

_KEEP_COLS = ["Entity_ID", "Seq_ID", "Auth_seq_ID", "Comp_ID",
              "Atom_ID", "Atom_type", "Val", "ChemShift_ID"]

class ChemicalShifts:
    """A tidy table of assigned chemical shifts (one row per atom)."""

    def __init__(self, data):
        self.data = data

    @classmethod
    def from_entry(cls, entry, reref=None, calc_csi=False):
        """Build from an NMRStarEntry's assigned_chemical_shifts saveframes."""
        frames = []
        for framecode, sf in entry.saveframe("assigned_chemical_shifts").items():
            cs = NMRStarEntry.loop_to_dataframe(sf["_Atom_chem_shift"])
            cs["name"] = sf.get("Name", ".")
            cs["ChemShift_ID"] = framecode
            cs = cls._clean(cs)
            frames.append(cs)

        df = pd.concat(frames, ignore_index=True)
        df["Seq_ID"] = df["Seq_ID"].astype(int)

        obj = cls(df)
        obj.entry = entry
        if reref in ("panav", "lacs"):
            obj.reref(method=reref)
        if calc_csi:
            obj.add_csi()
        return obj

    @classmethod
    def from_bmrb(cls, bmrb_id, reref=None, calc_csi=False, **fetch_kw):
        entry = NMRStarEntry.from_bmrb(bmrb_id, **fetch_kw)
        return cls.from_entry(entry, reref=reref, calc_csi=calc_csi)

    @staticmethod
    def _clean(df):
        df = df.copy()

        df = df[_KEEP_COLS]

        df["Entity_ID"] = pd.to_numeric(df["Entity_ID"], errors="coerce").astype("Int64")
        df["Seq_ID"] = pd.to_numeric(df["Seq_ID"], errors="coerce").astype("Int64")
        df["Auth_seq_ID"] = pd.to_numeric(df["Auth_seq_ID"], errors="coerce").astype("Int64")
        df["Val"] = pd.to_numeric(df["Val"].replace(".", np.nan), errors="coerce")

        str_cols = ["Comp_ID", "Atom_ID", "Atom_type", "ChemShift_ID"]
        df[str_cols] = df[str_cols].astype("string")

        return df

    # re-referencing

    def reref(self, method):
        """
        Re-reference shifts in place via the LACS/PANAV routine.
        """
        raise NotImplementedError()

    # chemical shift index

    _RANDOM_COIL = None

    @classmethod
    def _rc(cls, comp_id, atom_id):
        if cls._RANDOM_COIL is None:
            cls._RANDOM_COIL = get_random_coil()
        try:
            return cls._RANDOM_COIL[comp_id.upper()][atom_id.upper()]
        except KeyError:
            return np.nan

    @classmethod
    def _secondary_shift(cls, row):
        rc, val = cls._rc(row["Comp_ID"], row["Atom_ID"]), row["Val"]
        if rc is None or np.isnan(rc) or np.isnan(val):
            return np.nan
        return val - rc

    def _csi_raw(self, row, strict=False):
        """(CA - CB) secondary shift for one residue; CA-only fallback."""
        res = self.data.loc[self.data["Seq_ID"] == row["Seq_ID"]]
        ca = res[res["Atom_ID"] == "CA"]
        cb = res[res["Atom_ID"] == "CB"]
        ca_sec = self._secondary_shift(ca.iloc[0]) if len(ca) else np.nan
        cb_sec = self._secondary_shift(cb.iloc[0]) if len(cb) else np.nan
        if np.isfinite(ca_sec) and np.isfinite(cb_sec):
            return ca_sec - cb_sec
        if not strict and np.isfinite(ca_sec):
            return ca_sec
        return np.nan

    @staticmethod
    def _csi_index(value, comp_id, helix=0.7, strand=-0.7, gly=0.7):
        if np.isnan(value):
            return np.nan
        if value >= helix:
            return 1.0
        if value <= strand:
            return -1.0
        if comp_id == "GLY":
            if value >= gly:
                return 1.0
            if value <= -gly:
                return -1.0
        return 0.0

    def add_csi(self):
        """Add ``csi_raw`` and ``csi`` columns in place; returns self."""
        atoms = self.data["Atom_ID"].unique()
        if "CA" not in atoms or "CB" not in atoms:
            warnings.warn("CA and/or CB missing from Atom_ID; cannot calculate CSI",
                          UserWarning)
            return self
        self.data["csi_raw"] = self.data.apply(self._csi_raw, axis=1)
        self.data["csi"] = self.data.apply(
            lambda r: self._csi_index(r["csi_raw"], r["Comp_ID"]), axis=1
        ).astype(float)
        return self

    def __repr__(self):
        n = self.data["Seq_ID"].nunique() if len(self.data) else 0
        return f"ChemicalShifts(atoms={len(self.data)}, residues={n})"

    def peaklist(self, cs_saveframe=None, entity_id=None):
        from .peaklist import PeakList

        """Project these shifts to a backbone-amide PeakList (one H/N per residue)."""
        return PeakList.from_chemshifts(self, cs_saveframe=cs_saveframe, entity_id=entity_id)

    def get_entry(self):
        return self.entry.get_entry()

    def sequences(self, entity_id=None):
        return self.entry.sequences(entity_id=entity_id)