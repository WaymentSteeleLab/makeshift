"""
RelaxDB-style cleaning of deposited backbone relaxation data.

Assembles per-residue R1, R2, and heteronuclear NOE from a BMRB entry's
deposited relaxation saveframes, aligns them to the sequence, forms the R2/R1
ratio, and labels each residue by motional regime against a HYDRONMR rigid-body
prediction — following the RelaxDB curation in Wayment-Steele, El Nesr et al.
("Learning millisecond protein dynamics from what is missing in NMR spectra").

Per-residue label tokens:
    A  ordered / no detected motion
    ^  us-ms exchange  (R2/R1 elevated above the rigid prediction)
    v  ps-ns motion    (hetNOE <= cutoff)
    b  both ^ and v
    .  peak missing     (in sequence but no relaxation data)
    t  disordered terminus (outside the HYDRONMR-modeled region)
    p  proline          (no amide H)

The R2/R1 observable cancels the overall tumbling rate, so a single scaled
rigid prediction (HYDRONMR ``T1_over_T2``) is comparable across residues.
"""
import warnings

import numpy as np
import pandas as pd

from ..entry import NMRStarEntry
from ..utils.structures import detect_source

ORDERED, REX, PSNS, BOTH = "A", "^", "v", "b"
MISSING, TERMINUS, PROLINE = ".", "t", "p"



def _to_rate(values, units, kind):
    """
    Return relaxation values as rates (s^-1).
    """
    v = pd.to_numeric(values, errors="coerce").astype(float)
    u = (units or "").strip().lower().replace(" ", "").replace("^", "")
    if u in ("s-1", "s1", "1/s", "hz", "/s"):
        return v
    if u in ("s", "sec", "second", "seconds", "msec", "ms"):
        scale = 1e-3 if u in ("msec", "ms") else 1.0
        return 1.0 / (v * scale)
    # no usable units tag: guess. R2 rates are tens of s^-1; T2 times are <1 s.
    if np.isnan(v).all():
        return v  # absent/empty list: nothing to convert or warn about
    looks_like_rate = np.nanmedian(v) > (3.0 if kind == "T2" else 1.5)
    warnings.warn(
        f"{kind} list has no recognized units tag; assuming values are "
        f"{'rates (s^-1)' if looks_like_rate else 'times (s)'}. "
        "Pass the entry through with explicit units to avoid this guess.",
        UserWarning,
    )
    return v if looks_like_rate else 1.0 / v
 
 
class RelaxationProfile:
    """
    Per-residue R1/R2/hetNOE for one entity, aligned to its sequence, with
    RelaxDB motional labels.
 
    Attributes
    ----------
    table : DataFrame
        One row per sequence position (1-indexed). Columns: Seq_ID, residue,
        R1, R1_err, R2, R2_err, NOE, NOE_err, R2_R1, R2_R1_err, has_data
        (plus scaled_R2_R1_pred and label once those steps are run).
    sequence : str
        One-letter sequence the data is aligned to.
    entry_id, entity_id : identifiers carried for reference.
    """
 
    def __init__(self, table, sequence, entry_id=None, entity_id=None, entry=None):
        self.table = table
        self.sequence = sequence
        self.entry_id = entry_id
        self.entity_id = entity_id
        self.entry = entry
        self.scale_factor = None
 
    # construction
 
    @classmethod
    def from_bmrb(cls, bmrb_id, entity_id=None, sequence=None, **fetch_kw):
        """Fetch a BMRB entry and build a profile from its deposited relaxation."""
        entry = NMRStarEntry.from_bmrb(bmrb_id, **fetch_kw)
        return cls.from_entry(entry, entity_id=entity_id, sequence=sequence)
 
    @classmethod
    def from_entry(cls, entry, entity_id=None, sequence=None):
        """
        Build from an already-parsed NMRStarEntry. 
        Pulls R1 (from T1), R2 (from T2), and hetNOE, converting times to rates using each list's units tag.
        """
        if sequence is None:
            seqs = entry.sequences()
            if entity_id is not None:
                sequence = entry.sequences(entity_id=entity_id)
            elif len(seqs):
                sequence = seqs["Polymer_seq_one_letter_code"].iloc[0]
                entity_id = seqs["ID"].iloc[0]
            if not isinstance(sequence, str) or not sequence:
                raise ValueError(
 
                )
 
        n = len(sequence)
        table = pd.DataFrame({
            "Seq_ID": np.arange(1, n + 1),
            "residue": list(sequence),
        })
 
        # R1 and R2 come from the T1 / T2 lists (converted to rates)
        for kind, col in (("T1", "R1"), ("T2", "R2")):
            df = entry.relaxation(kind)
            units = cls._list_units(entry, kind)
            table[col], table[f"{col}_err"] = cls._align_rate(
                df, n, units, kind)
 
        # hetNOE is already a ratio
        noe = entry.relaxation("NOE")
        table["NOE"], table["NOE_err"] = cls._align_plain(noe, n)
 
        table["R2_R1"] = table["R2"] / table["R1"]
        table["R2_R1_err"] = table["R2_R1"] * np.sqrt(
            (table["R2_err"] / table["R2"]) ** 2
            + (table["R1_err"] / table["R1"]) ** 2)
        table["has_data"] = table[["R1", "R2"]].notna().all(axis=1)
 
        return cls(table, sequence,
                   entry_id=getattr(entry, "entry_id", None),
                   entity_id=entity_id, entry=entry)
 
    # alignment helpers
 
    @staticmethod
    def _list_units(entry, kind):
        for sf in entry.saveframe(f"heteronucl_{kind}_relaxation").values():
            u = sf.get(f"{kind}_val_units")
            if u and u not in (".", "?"):
                return u
        return None
 
    @staticmethod
    def _series_by_seqid(df, n, value_col="Val", err_col="Val_err"):
        """Index a relaxation DataFrame's value/err onto 1..n by Seq_ID."""
        val = pd.Series(np.nan, index=np.arange(1, n + 1))
        err = pd.Series(np.nan, index=np.arange(1, n + 1))
        if df is None or df.empty:
            return val, err
        sub = df.dropna(subset=["Seq_ID"])
        for _, row in sub.iterrows():
            s = int(row["Seq_ID"])
            if 1 <= s <= n:
                val.loc[s] = row.get(value_col, np.nan)
                err.loc[s] = row.get(err_col, np.nan)
        return val, err
 
    @classmethod
    def _align_rate(cls, df, n, units, kind):
        val, err = cls._series_by_seqid(df, n)
        rate = _to_rate(val.values, units, kind)
        # convert a time error into a rate error: d(1/T) = dT / T^2
        is_time = not (rate.size and np.allclose(rate, val.values, equal_nan=True))
        rate_err = (err.values / val.values ** 2) if is_time else err.values
        return rate, rate_err
 
    @classmethod
    def _align_plain(cls, df, n):
        val, err = cls._series_by_seqid(df, n)
        return val.values, err.values
 
    # rigid-body prediction
 
    def add_rigid_prediction(self, pdb=None, source="auto", config=None,
                             chain=None, noe_cut=0.65, **fetch_kw):
        """
        Run HYDRONMR on a structure and scale its rigid R2/R1 (T1_over_T2) to the
        observed data, so elevated R2/R1 stands out as exchange.
 
        `pdb` may be a local file, a 4-character PDB id (fetched from RCSB), or a
        UniProt accession (fetched from AlphaFold DB) — pass `source=` to force
        one. 
        
        If `pdb` is None, the entry's own deposited PDB code is used when it
        cites one; otherwise this raises (makeshift does not predict structure).
 
        The scale factor is fit by least squares on ordered residues (hetNOE
        above `noe_cut` where available), mirroring classify.fit_R2_rigid. Adds
        `scaled_R2_R1_pred` and `NOE_pred`; residues outside the modeled region
        keep NaN.
        """
        from ..hydronmr import run as run_hydronmr
        from ..utils.structures import fetch_structure
 
        if pdb is None:
            pdb_ids = self.entry.get_pdb_ids() if self.entry is not None else []
            af_ids = self.entry.get_alphafold_ids() if self.entry is not None else []
            if source == "rcsb":
                if not pdb_ids:
                    raise ValueError("entry cites no PDB; pass pdb=<PDB id | path>")
                pdb = pdb_ids[0]
            elif source == "afdb":
                if not af_ids:
                    raise ValueError("entry cites no AlphaFold/UniProt accession; "
                                     "pass pdb=<UniProt accession | path>")
                pdb = af_ids[0]
            elif pdb_ids:                    # source == "auto": prefer deposited PDB
                pdb, source = pdb_ids[0], "rcsb"
            elif af_ids:                     # else fall back to AlphaFold
                pdb, source = af_ids[0], "afdb"
            else:
                raise ValueError(
                    "no structure given and the entry cites no PDB or "
                    "AlphaFold/UniProt accession; pass pdb=<path | PDB id | "
                    "UniProt accession> (experimental or predicted) to enable "
                    "exchange labeling"
                )
            print(f"  no pdb given; using {source} structure {pdb}")
 
        pdb_path = fetch_structure(pdb, source=source, **fetch_kw)
        result = run_hydronmr(pdb_path, config_path=config) if config \
            else run_hydronmr(pdb_path)
        hydro = result.to_dataframe()
        if chain is not None:
            hydro = hydro[hydro["chain"] == chain]
        hydro = hydro.rename(columns={"seqpos": "Seq_ID"})
 
        t = self.table.merge(
            hydro[["Seq_ID", "T1_over_T2", "NOE"]].rename(
                columns={"T1_over_T2": "_pred_ratio", "NOE": "NOE_pred"}),
            on="Seq_ID", how="left")
 
        ordered = (t["_pred_ratio"].notna() & t["R2_R1"].notna()
                   & ((t["NOE"] > noe_cut) | t["NOE"].isna()))
        n_match = int((t["_pred_ratio"].notna() & t["R2_R1"].notna()).sum())
        pred = t.loc[ordered, "_pred_ratio"]
        obs = t.loc[ordered, "R2_R1"]
        self.scale_factor = float((obs * pred).sum() / (pred ** 2).sum())
 
        t["scaled_R2_R1_pred"] = self.scale_factor * t["_pred_ratio"]
        t = t.drop(columns="_pred_ratio")
        self.table = t
        print(f"  HYDRONMR: {n_match} residues matched structure to data, "
              f"scale factor {self.scale_factor:.3f} "
              f"({int(ordered.sum())} ordered residues used)")
        return self
 
    # labeling
 
    def label(self, rex_n_std=1.0, noe_cut=0.65):
        """
        Assign a label token to every residue and return the label string.
 
        Requires `add_rigid_prediction` first for the exchange (`^`) call, which
        flags residues whose R2/R1 exceeds the rigid prediction by more than
        `rex_n_std` standard deviations of that excess across modeled residues.
        ps-ns motion (`v`) is hetNOE <= `noe_cut`.
        """
        t = self.table
        have_pred = "scaled_R2_R1_pred" in t.columns
 
        excess = pd.Series(np.nan, index=t.index)
        rex_mask = pd.Series(False, index=t.index)
        if have_pred:
            modeled = t["scaled_R2_R1_pred"].notna() & t["R2_R1"].notna()
            excess[modeled] = t.loc[modeled, "R2_R1"] - t.loc[modeled, "scaled_R2_R1_pred"]
            thresh = excess[modeled].mean() + rex_n_std * excess[modeled].std()
            rex_mask = excess > thresh
        else:
            warnings.warn(
                "no rigid prediction set; exchange (^) cannot be called. "
                "trying to run add_rigid_prediction(pdb) first with default "
                "parameters .", UserWarning)
 
        psns_mask = t["NOE"] <= noe_cut
 
        labels = []
        for i, row in t.iterrows():
            if row["residue"] == "P":
                labels.append(PROLINE)
            elif not row["has_data"]:
                # outside the modeled region -> terminus; else simply missing
                if have_pred and pd.isna(row.get("scaled_R2_R1_pred")):
                    labels.append(TERMINUS)
                else:
                    labels.append(MISSING)
            else:
                rex = bool(rex_mask.get(i, False))
                psns = bool(psns_mask.get(i, False))
                labels.append(BOTH if (rex and psns) else
                              REX if rex else
                              PSNS if psns else ORDERED)
 
        t["label"] = labels
        self.table = t
        return "".join(labels)
 
    @property
    def label_string(self):
        if "label" not in self.table.columns:
            return None
        return "".join(self.table["label"])
 
    # plotting
 
    def plot(self, data_type="R2_R1", ax=None, figsize=(6, 1.5)):
        """
        Plot a relaxation observable along the sequence with 
        motion labels:
            orange = exchange (^/b)
            blue = ps-ns (v)
            black = ordered (A)
            purple P = proline
            red star = missing 
        The scaled rigid prediction is overlaid for R2_R1. 
        Requires `label()` first.
        """
        import matplotlib.pyplot as plt
 
        t = self.table
        if "label" not in t.columns:
            raise ValueError("call label() before plot()")
        if data_type not in t.columns:
            raise ValueError(f"no column {data_type!r} in table")
 
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        x = t["Seq_ID"].values
        y = t[data_type].values
        ax.plot(x, y, color="black", lw=0.5, zorder=5)
 
        if data_type == "R2_R1" and "scaled_R2_R1_pred" in t.columns:
            ax.plot(x, t["scaled_R2_R1_pred"].values, color="grey", zorder=5)
 
        ymin, ymax = ax.get_ylim()
        p_pos = ymin + 0.05 * (ymax - ymin)
        star_pos = ymin + 0.9 * (ymax - ymin)
        err = t.get(f"{data_type}_err")
 
        colors = {REX: "tab:orange", BOTH: "tab:orange",
                  PSNS: "tab:blue", ORDERED: "black", TERMINUS: "grey"}
        for _, row in t.iterrows():
            j, lab = row["Seq_ID"], row["label"]
            if lab == PROLINE:
                ax.axvline(j, color="tab:purple", alpha=0.5, lw=0.5)
                ax.text(j - 0.5, p_pos, "P", color="tab:purple",
                        fontsize=5, weight="bold")
            elif lab == MISSING:
                ax.axvline(j, color="tab:red", alpha=0.5, lw=0.5)
                ax.scatter([j], [star_pos], marker="*", color="tab:red")
            elif lab in colors and lab != TERMINUS and pd.notna(row[data_type]):
                e = err.loc[row.name] if err is not None else None
                ax.errorbar(j, row[data_type], yerr=e, fmt=".",
                            color=colors[lab], zorder=10)
        ax.set_xlim(0, len(self.sequence) + 1)
        ax.set_xlabel("Residue")
        return ax
 
    def __repr__(self):
        n = int(self.table["has_data"].sum()) if "has_data" in self.table else 0
        return (f"RelaxationProfile(entry_id={self.entry_id!r}, "
                f"residues={len(self.sequence)}, with_data={n})")