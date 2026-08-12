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
        self.entry = None

    @classmethod
    def from_entry(cls, entry, reref=None, calc_csi=False):
        """Build from an NMRStarEntry's assigned_chemical_shifts saveframes."""
        frames = []
        for framecode, sf in entry.saveframe("assigned_chemical_shifts").items():
            loop = sf.get("_Atom_chem_shift")
            if not loop:
                continue
            cs = NMRStarEntry.loop_to_dataframe(loop)
            cs["name"] = sf.get("Name", ".")
            cs["ChemShift_ID"] = framecode
            cs = cls._clean(cs)
            sids = [r.get("Sample_ID") for r in sf.get("_Chem_shift_experiment", [])]
            sids = [s for s in sids if s and s not in (".", "?")]
            cs["Sample_ID"] = ",".join(dict.fromkeys(sids)) or pd.NA
            frames.append(cs)

        if not frames:
            raise ValueError(
                f"entry {entry.entry_id!r} has no assigned chemical shifts "
                "to build a ChemicalShifts from."
            )

        df = pd.concat(frames, ignore_index=True)
        df["Seq_ID"] = df["Seq_ID"].astype(int)

        obj = cls(df)
        obj.entry = entry
        if reref in ("panav", "lacs"):
            obj.reref(method=reref)
        if calc_csi:
            method = calc_csi if isinstance(calc_csi, str) else "wishart_94"
            obj.add_csi(method=method)
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
        Re-reference shifts in place via the LACS or PANAV routine.

        For ``method="panav"``, also stores a CONA fragment-scan summary on
        ``self.reref_cona`` (3–6 residue window confirmation scores).
        """
        from .reref import compute_offsets, apply_offsets

        offsets, check, meta = compute_offsets(self.data, method)
        self.reref_method = method
        self.reref_offsets = offsets
        self.reref_check = check
        self.reref_cona = meta

        if not offsets or not any(v is not None for v in offsets.values()):
            warnings.warn(
                f"{method} re-referencing produced no offsets "
                "(insufficient backbone shifts or all fits failed); "
                "shifts left unchanged.",
                UserWarning,
            )
            return self

        self.data = apply_offsets(self.data, offsets)
        return self

    # chemical shift index (Wishart); CA−CB helpers below also feed LACS

    _RANDOM_COIL = None
    _CSI_METHODS = {
        "wishart_92": ("HA",),
        "wishart_94": None,          # all nuclei + consensus
        "wishart_94_ha": ("HA",),
        "wishart_94_ca": ("CA",),
        "wishart_94_cb": ("CB",),
        "wishart_94_c": ("C",),
    }

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
        if np.isnan(rc) or np.isnan(val):
            return np.nan
        return val - rc

    def _csi_raw(self, row, strict=False):
        """(CA − CB) secondary shift per residue — used by LACS fitting."""
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

    def add_csi(self, method="wishart_94", assign_ss=True):
        """
        Add Wishart chemical-shift index columns in place; returns self.

        Parameters
        ----------
        method : str
            ``'wishart_92'`` — 1Ha CSI (Wishart, Sykes & Richards 1992).

            ``'wishart_94'`` (default) — HA + CA + CB + C' indices with
            density-filtered secondary structure and majority-rules consensus
            (Wishart & Sykes 1994).

            ``'wishart_94_ha'`` / ``'_ca'`` / ``'_cb'`` / ``'_c'`` — single-nucleus
            1994 protocol (CB is strand-only).
        assign_ss : bool
            Run the stage-2 density filter (and consensus for ``wishart_94``).

        Notes
        -----
        Ternary signs: HA +1 = strand / -1 = helix; CA and C' +1 = helix /
        -1 = strand; CB +1 = strand only. Consensus ``ss`` is H / E / C;
        ``csi`` mirrors that as +1 / -1 / 0. Per-residue detail is on
        ``self.csi_table``.

        LACS re-referencing does **not** use this method — it computes its own
        continuous CA-CB secondary shift via ``_csi_raw``.
        """
        method = str(method).lower()
        if method not in self._CSI_METHODS:
            raise ValueError(
                f"unknown CSI method {method!r}; "
                f"choose from {tuple(self._CSI_METHODS)}"
            )
        self.csi_method = method

        from . import csi as wishart_csi

        atoms = self._CSI_METHODS[method]
        if atoms is None:
            atoms = wishart_csi._WISHART_ATOMS
        table = wishart_csi.wishart_table(
            self.data, atoms=atoms, assign_ss=assign_ss
        )
        self.csi_table = table
        by_seq = table.set_index("Seq_ID")

        if method == "wishart_94":
            for col in ("HA", "CA", "CB", "C"):
                if col in by_seq.columns:
                    self.data[f"csi_{col.lower()}"] = self.data["Seq_ID"].map(
                        by_seq[col]
                    )
            if "ss" in by_seq.columns:
                self.data["ss"] = self.data["Seq_ID"].map(by_seq["ss"])
            if "csi" in by_seq.columns:
                self.data["csi"] = self.data["Seq_ID"].map(by_seq["csi"])
            if "CA_raw" in by_seq.columns:
                self.data["csi_raw"] = self.data["Seq_ID"].map(by_seq["CA_raw"])
        else:
            atom = atoms[0]
            self.data["csi"] = self.data["Seq_ID"].map(by_seq[atom])
            raw_col = f"{atom}_raw"
            self.data["csi_raw"] = (
                self.data["Seq_ID"].map(by_seq[raw_col])
                if raw_col in by_seq.columns else np.nan
            )
            if assign_ss and "ss" in by_seq.columns:
                self.data["ss"] = self.data["Seq_ID"].map(by_seq["ss"])
        return self

    def __repr__(self):
        n = self.data["Seq_ID"].nunique() if len(self.data) else 0
        return f"ChemicalShifts(atoms={len(self.data)}, residues={n})"

    def peaklist(self, cs_saveframe=None, entity_id=None, dims=None):
        from .peaklist import PeakList

        """Project these shifts to a PeakList (backbone amide H/N by default;
        pass ``dims`` for other atom combinations)."""
        return PeakList.from_chemshifts(self, cs_saveframe=cs_saveframe, entity_id=entity_id, dims=dims)

    def get_entry(self):
        return self.entry.get_entry()

    def sequences(self, entity_id=None):
        return self.entry.sequences(entity_id=entity_id)