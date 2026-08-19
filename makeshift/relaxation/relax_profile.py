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
    .  peak missing     (no amide peak assigned for this residue — i.e. the
                         peak is absent from the spectrum)
    p  proline          (no amide H)
    x  no data          (an amide peak is assigned, but no relaxation data was
                         measured/reported for it)

The R2/R1 observable cancels the overall tumbling rate, so a single scaled
rigid prediction (HYDRONMR ``T1_over_T2``) is comparable across residues.
"""
import warnings

import numpy as np
import pandas as pd

from ..entry import NMRStarEntry
from ..utils.structures import detect_source
from . import fixes

ORDERED, REX, PSNS, BOTH = "A", "^", "v", "b"
MISSING, TERMINUS, PROLINE, NODATA = ".", "t", "p", "x"

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
    tau_scale : the calibrated diffusion-tensor timescale correction (see
        `add_rigid_prediction`/`fit_order_parameters`), None until one of
        those has run.
    """
 
    def __init__(self, table, sequence, entry_id=None, entity_id=None, entry=None):
        self.table = table
        self.sequence = sequence
        self.entry_id = entry_id
        self.entity_id = entity_id
        self.entry = entry
        self.scale_factor = None
        self.tau_scale = None
 
    # construction
 
    @classmethod
    def from_bmrb(cls, bmrb_id, entity_id=None, sequence=None, peaklist=None,
                  **fetch_kw):
        """Fetch a BMRB entry and build a profile from its deposited relaxation."""
        entry = NMRStarEntry.from_bmrb(bmrb_id, **fetch_kw)
        return cls.from_entry(entry, entity_id=entity_id, sequence=sequence,
                              peaklist=peaklist)
 
    @classmethod
    def from_entry(cls, entry, entity_id=None, sequence=None, peaklist=None):
        """
        Build from an already-parsed NMRStarEntry.
        Pulls R1 (from T1), R2 (from T2), and hetNOE, converting times to rates
        using each list's units tag.

        `peaklist` (a PeakList, a DataFrame with a Seq_ID column, or an iterable
        of Seq_IDs) marks which residues have an assigned amide peak; if None,
        the entry's own backbone amide peaks are derived via PeakList. Residues
        with no assigned peak are flagged "missing" (.); residues with a peak but
        no relaxation data are flagged "no data" (x).
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
                    "could not resolve a sequence; pass sequence=... explicitly"
                )
 
        n = len(sequence)
        table = pd.DataFrame({
            "Seq_ID": np.arange(1, n + 1),
            "residue": list(sequence),
        })
 
        for kind, col in (("T1", "R1"), ("T2", "R2")):
            df = entry.relaxation(kind)
            units = cls._list_units(entry, kind)
            table[col], table[f"{col}_err"] = cls._align_rate(
                df, n, units, kind, entry_id=entry.entry_id)
 
        noe = entry.relaxation("NOE")
        table["NOE"], table["NOE_err"] = cls._align_plain(noe, n)
 
        table["R2_R1"] = table["R2"] / table["R1"]
        table["R2_R1_err"] = table["R2_R1"] * np.sqrt(
            (table["R2_err"] / table["R2"]) ** 2
            + (table["R1_err"] / table["R1"]) ** 2)
        table["has_data"] = table[["R1", "R2"]].notna().all(axis=1)

        if peaklist is None:
            peaklist = cls._entry_peaklist(entry, entity_id)
        present = cls._peaklist_seqids(peaklist)
        table["has_HN"] = table["Seq_ID"].isin(present)
 
        return cls(table, sequence,
                   entry_id=getattr(entry, "entry_id", None),
                   entity_id=entity_id, entry=entry)
 
    # alignment helpers
 
    @staticmethod
    def _list_units(entry, kind):
        override = fixes.unit_override(entry.entry_id, kind)
        if override is not None:
            return override
        for sf in entry.saveframe(f"heteronucl_{kind}_relaxation").values():
            u = sf.get(f"{kind}_val_units")
            if u and u not in (".", "?"):
                return u
        return None

    @staticmethod
    def _series_by_seqid(df, n, value_col="Val", err_col="Val_err"):
        """
        Index a relaxation DataFrame's value/err onto 1..n by Seq_ID.

        Restricts to backbone N/H rows when Atom_ID is present and any are
        found, keeping the first match per residue — some entries (e.g. for
        Arg, Trp) also deposit a sidechain relaxation value under the same
        Seq_ID, which would otherwise silently overwrite the backbone one.
        """
        val = pd.Series(np.nan, index=np.arange(1, n + 1))
        err = pd.Series(np.nan, index=np.arange(1, n + 1))
        if df is None or df.empty:
            return val, err
        sub = df.dropna(subset=["Seq_ID"])
        if "Atom_ID" in sub.columns:
            backbone = sub["Atom_ID"].isin(("N", "H"))
            if backbone.any():
                sub = sub[backbone]
        sub = sub.drop_duplicates(subset="Seq_ID", keep="first")
        for _, row in sub.iterrows():
            s = int(row["Seq_ID"])
            if 1 <= s <= n:
                val.loc[s] = row.get(value_col, np.nan)
                err.loc[s] = row.get(err_col, np.nan)
        return val, err

    @classmethod
    def _align_rate(cls, df, n, units, kind, entry_id=None):
        value_col, err_col = "Val", "Val_err"
        if fixes.value_err_swapped(entry_id, kind):
            value_col, err_col = err_col, value_col
        val, err = cls._series_by_seqid(df, n, value_col=value_col, err_col=err_col)
        rate = _to_rate(val.values, units, kind)
        # convert a time error into a rate error: d(1/T) = dT / T^2
        is_time = not (rate.size and np.allclose(rate, val.values, equal_nan=True))
        rate_err = (err.values / val.values ** 2) if is_time else err.values

        if fixes.err_reciprocal(entry_id, kind):
            rate_err = 1.0 / rate_err
        if fixes.err_invalid(entry_id, kind):
            rate_err = np.full(n, np.nan)
        return rate, rate_err
 
    @classmethod
    def _align_plain(cls, df, n):
        val, err = cls._series_by_seqid(df, n)
        return val.values, err.values

    # chemical-shift (amide peak) availability

    @staticmethod
    def _entry_peaklist(entry, entity_id):
        """
        The entry's backbone amide peak list for one entity (residues with both
        an amide H and N shift), via PeakList — reused, not reimplemented.
        """
        try:
            from ..peaklist import PeakList
            return PeakList.from_entry(entry, entity_id=entity_id)
        except Exception:
            return None

    @staticmethod
    def _peaklist_seqids(peaklist):
        """
        Seq_IDs present in a peaklist. Accepts a PeakList, a DataFrame with a
        Seq_ID column (one row per H-N peak), or a plain iterable of Seq_IDs.
        Rows are required to have both H_ppm and N_ppm where those columns exist.
        """
        if peaklist is None:
            return set()
        data = getattr(peaklist, "data", peaklist)   # PeakList -> its .data
        out = set()
        if isinstance(data, pd.DataFrame):
            if "Seq_ID" not in data.columns:
                return set()
            sub = data.dropna(subset=["Seq_ID"])
            ppm_cols = [c for c in ("H_ppm", "N_ppm") if c in sub.columns]
            if ppm_cols:
                sub = sub.dropna(subset=ppm_cols)
            values = sub["Seq_ID"]
        else:
            values = data                            # iterable of Seq_IDs
        for s in values:
            try:
                out.add(int(s))
            except (TypeError, ValueError):
                continue
        return out
 
    # structure resolution (shared by add_rigid_prediction and
    # fit_order_parameters)

    def _resolve_pdb(self, pdb, source, **fetch_kw):
        """
        Resolve `pdb` to a local structure file path. `pdb` may be a local
        file, a 4-character PDB id (fetched from RCSB), or a UniProt
        accession (fetched from AlphaFold DB) — pass `source=` to force
        one. If `pdb` is None, the entry's own deposited PDB code is used
        when it cites one; otherwise this raises (makeshift does not
        predict structure).
        """
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
                    "UniProt accession> (experimental or predicted)"
                )
            print(f"  no pdb given; using {source} structure {pdb}")

        return fetch_structure(pdb, source=source, **fetch_kw)

    def _detect_field_mhz(self):
        """
        The entry's 1H spectrometer frequency (MHz), read from its own
        heteronucl_T1_relaxation saveframe (`Spectrometer_frequency_1H`).
        Returns None if unavailable.
        """
        if self.entry is None:
            return None
        try:
            sfs = self.entry.saveframe("heteronucl_T1_relaxation")
        except Exception:
            return None
        for sf in sfs.values():
            val = sf.get("Spectrometer_frequency_1H")
            if val not in (None, ".", "?"):
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return None

    def _residue_geometry(self, g, pdb_path, chain=None):
        """
        For every residue with data and a matched N-H bond vector: its
        Woessner mode amplitudes/taus (`mode_amplitudes`, unscaled) and
        its table row. Returns {position -> (amplitudes, taus, row)},
        shared by `add_rigid_prediction` and `fit_order_parameters` so
        both work from identical per-residue tumbling geometry.
        """
        from ..hydronmr.physics.nmr import mode_amplitudes
        from ..hydronmr.physics.pdb import nh_bond_vectors, parse_pdb_atoms

        nh_vectors = nh_bond_vectors(parse_pdb_atoms(pdb_path))
        if chain is not None:
            nh_vectors = {k: v for k, v in nh_vectors.items() if k[0] == chain}
        vec_by_seqid = {}
        for (c, resseq), v in nh_vectors.items():
            vec_by_seqid.setdefault(resseq, v)

        geometry = {}
        for pos, (_, row) in enumerate(self.table.iterrows()):
            if not row.get("has_data", False):
                continue
            v = vec_by_seqid.get(int(row["Seq_ID"]))
            if v is None:
                continue
            geometry[pos] = (*mode_amplitudes(g, v), row)
        return geometry

    def _calibration_subset(self, geometry, noe_cut):
        """
        (amplitudes, taus, R1, R1_err, R2, R2_err) tuples for the
        presumed-rigid residues (NOE > `noe_cut`) used to calibrate
        `model_free.calibrate_tau_scale`; falls back to every residue
        with R1/R2 data (with a warning) if fewer than 3 qualify.
        """
        def usable(row):
            return (row.get("R1") is not None and np.isfinite(row.get("R1"))
                    and row.get("R2") is not None and np.isfinite(row.get("R2")))

        calibration = [
            (amplitudes, taus, row.get("R1"), row.get("R1_err"),
             row.get("R2"), row.get("R2_err"))
            for amplitudes, taus, row in geometry.values()
            if usable(row) and row.get("NOE") is not None
            and np.isfinite(row.get("NOE")) and row.get("NOE") > noe_cut
        ]
        if len(calibration) < 3:
            calibration = [
                (amplitudes, taus, row.get("R1"), row.get("R1_err"),
                 row.get("R2"), row.get("R2_err"))
                for amplitudes, taus, row in geometry.values() if usable(row)
            ]
            warnings.warn(
                f"fewer than 3 residues with NOE > {noe_cut}; calibrating "
                "the diffusion-tensor timescale against all residues with "
                "R1/R2 data instead (less reliable if many are flexible).",
                UserWarning,
            )
        return calibration

    # rigid-body prediction
 
    def add_rigid_prediction(self, pdb=None, source="auto", config=None,
                             chain=None, noe_cut=0.65, field_mhz=None,
                             **fetch_kw):
        """
        Run HYDRONMR on a structure to get a rigid-body R2/R1 (T1_over_T2)
        and NOE prediction per residue, so elevated R2/R1 stands out as
        exchange.

        `pdb` may be a local file, a 4-character PDB id (fetched from RCSB), or a
        UniProt accession (fetched from AlphaFold DB) — pass `source=` to force
        one.

        If `pdb` is None, the entry's own deposited PDB code is used when it
        cites one; otherwise this raises (makeshift does not predict structure).

        Before predicting, a single global timescale correction (all five
        Woessner modes' tau_k -> k*tau_k) is calibrated against the
        presumed-rigid subset (NOE > `noe_cut`) via
        `model_free.calibrate_tau_scale` — the same calibration
        `fit_order_parameters` uses, so both methods agree on one rigid-body
        baseline rather than each fitting an independent correction. (The
        bead model's diffusion-tensor anisotropy *shape* is validated
        elsewhere, demos/hydronmr_validation; its absolute timescale is a
        known approximation — see makeshift/hydronmr/physics/structure.py.)
        A small residual multiplicative correction on the ratio specifically
        (`self.scale_factor`, expected close to 1.0 now that the timescale
        itself is calibrated) is still fit on top, mirroring the previous
        behavior and absorbing whatever the timescale-only correction
        doesn't reach.

        `field_mhz` is the 1H spectrometer frequency to predict at; if not
        given, it's read from the entry's own heteronucl_T1_relaxation
        saveframe when available, else the structure's rigid prediction
        falls back to HYDRONMR's own default config field.

        Adds `scaled_R2_R1_pred` and `NOE_pred`; residues outside the
        modeled region keep NaN. Sets `self.tau_scale` and
        `self.scale_factor`.
        """
        from ..hydronmr import run as run_hydronmr
        from ..hydronmr.physics.nmr import dipolesnmr
        from . import model_free

        pdb_path = self._resolve_pdb(pdb, source, **fetch_kw)
        result = run_hydronmr(pdb_path, config_path=config) if config \
            else run_hydronmr(pdb_path)
        g = result.state

        if field_mhz is None:
            field_mhz = self._detect_field_mhz()
        if field_mhz is not None:
            b0_tesla = 2.0 * np.pi * field_mhz * 1.0e6 / abs(g.gamma_h)
            dipolesnmr(g, b0_tesla=b0_tesla, gamma_h=g.gamma_h, gamma_x=g.gamma_x,
                      r_nh_angstrom=g.r_nh * 1.0e10, csa_ppm=g.csa * 1.0e6)

        geometry = self._residue_geometry(g, pdb_path, chain=chain)
        calibration = self._calibration_subset(geometry, noe_cut)
        self.tau_scale = model_free.calibrate_tau_scale(
            calibration, g.d2, g.c2, g.gamma_h, g.gamma_x, g.omega_h, g.omega_x)

        n = len(self.table)
        ratio_pred = np.full(n, np.nan)
        noe_pred = np.full(n, np.nan)
        for pos, (amplitudes, taus, row) in geometry.items():
            taus_scaled = [tau * self.tau_scale for tau in taus]
            r1p, r2p, noep = model_free.relaxation_rates(
                g.d2, g.c2, g.gamma_h, g.gamma_x, g.omega_h, g.omega_x,
                amplitudes, taus_scaled, S2=1.0, tau_e=0.0)
            ratio_pred[pos] = r2p / r1p
            noe_pred[pos] = noep

        t = self.table.copy()
        t["_pred_ratio"] = ratio_pred
        t["NOE_pred"] = noe_pred

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
              f"tau_scale {self.tau_scale:.3f}, residual ratio scale "
              f"{self.scale_factor:.3f} ({int(ordered.sum())} ordered residues used)")
        return self

    # model-free order parameters

    def fit_order_parameters(self, pdb=None, source="auto", chain=None,
                             field_mhz=None, config=None, sigma_flag=2.0,
                             noe_cut=0.65, **fetch_kw):
        """
        Fit per-residue S^2 (Models 1-3: S2 alone, S2+tau_e, S2+Rex) from
        this profile's R1/R2/NOE, reusing the same anisotropic rotational
        diffusion tensor and per-residue Woessner mode decomposition
        already used for the HYDRONMR rigid-body prediction (see
        `add_rigid_prediction`) -- internal motion is layered on top of
        that existing per-residue anisotropic-tumbling spectral density
        rather than assuming isotropic tumbling.

        Before fitting, a single global timescale correction (all five
        modes' tau_k -> k*tau_k, same k for every residue) is calibrated
        against the presumed-rigid subset (NOE > `noe_cut`) via
        `model_free.calibrate_tau_scale`. The bead model's tumbling
        *anisotropy shape* is validated elsewhere (demos/hydronmr_validation)
        to track the true diffusion tensor well, but its *absolute*
        timescale is a known approximation (see
        makeshift/hydronmr/physics/structure.py) -- `add_rigid_prediction`
        already corrects the analogous offset for its R2/R1 ratio; this
        does the same for R1 and NOE, which the ratio-only correction
        doesn't reach, before any residue's S2 is fit.

        Models 4/5 (S2+tau_e+Rex, or S2f+S2s+tau_e) are not fit: against
        only R1/R2/NOE at a single field they are exactly- or
        over-parameterized and, in practice, ill-conditioned. Model
        selection among 1/2/3 follows from which residual each extra
        parameter physically explains, not a nested F-test/SSE cascade:
        a NOE deficit relative to the parameter-free rigid-tumbling
        prediction implicates tau_e (-> Model 2, since NOE is exactly
        independent of S2 when tau_e=0); an R2 excess beyond what the
        R1-derived S2 already explains implicates exchange (-> Model 3).
        Residues showing both signals are outside this two-parameter
        scope and are labeled "ambiguous" rather than forced into either
        model.

        `pdb`/`source` are as in `add_rigid_prediction`. `field_mhz` is
        the 1H spectrometer frequency the relaxation data were recorded
        at; if not given, it's read from the entry's own
        heteronucl_T1_relaxation saveframe and this raises if that isn't
        available (pass it explicitly for data not sourced from
        `from_entry`/`from_bmrb`, or for entries lacking that tag).
        `sigma_flag` sets how many (error-propagated) standard deviations
        a NOE/R2 residual must exceed to flag tau_e/Rex.

        Adds columns: S2, S2_err, mf_model ("1"/"2"/"3"/"ambiguous"),
        tau_e_ps, Rex, NOE_pred_rigid, noe_flag, r2_flag. Residues with no
        R1 or no matched structural N-H vector keep mf_model=None / S2=NaN.
        Sets `self.tau_scale` to the calibrated timescale correction.
        """
        from ..hydronmr import run as run_hydronmr
        from ..hydronmr.physics.nmr import dipolesnmr
        from . import model_free

        pdb_path = self._resolve_pdb(pdb, source, **fetch_kw)
        result = run_hydronmr(pdb_path, config_path=config) if config \
            else run_hydronmr(pdb_path)
        g = result.state

        if field_mhz is None:
            field_mhz = self._detect_field_mhz()
            if field_mhz is None:
                raise ValueError(
                    "could not determine the spectrometer field from the "
                    "entry; pass field_mhz=<1H frequency in MHz> explicitly"
                )
        b0_tesla = 2.0 * np.pi * field_mhz * 1.0e6 / abs(g.gamma_h)
        dipolesnmr(g, b0_tesla=b0_tesla, gamma_h=g.gamma_h, gamma_x=g.gamma_x,
                  r_nh_angstrom=g.r_nh * 1.0e10, csa_ppm=g.csa * 1.0e6)

        geometry = self._residue_geometry(g, pdb_path, chain=chain)
        calibration = self._calibration_subset(geometry, noe_cut)
        self.tau_scale = model_free.calibrate_tau_scale(
            calibration, g.d2, g.c2, g.gamma_h, g.gamma_x, g.omega_h, g.omega_x)

        t = self.table.copy()
        n = len(t)
        S2 = np.full(n, np.nan)
        S2_err = np.full(n, np.nan)
        mf_model = np.array([None] * n, dtype=object)
        tau_e_ps = np.full(n, np.nan)
        Rex = np.full(n, np.nan)
        noe_pred_rigid = np.full(n, np.nan)
        noe_flag = np.zeros(n, dtype=bool)
        r2_flag = np.zeros(n, dtype=bool)

        for pos, (amplitudes, taus, row) in geometry.items():
            taus_scaled = [tau * self.tau_scale for tau in taus]
            fit = model_free.fit_residue(
                row.get("R1"), row.get("R1_err"),
                row.get("R2"), row.get("R2_err"),
                row.get("NOE"), row.get("NOE_err"),
                amplitudes, taus_scaled, g.d2, g.c2, g.gamma_h, g.gamma_x,
                g.omega_h, g.omega_x, sigma_flag=sigma_flag)
            if fit.model is None:
                continue
            S2[pos] = fit.S2
            S2_err[pos] = fit.S2_err
            mf_model[pos] = fit.model
            tau_e_ps[pos] = fit.tau_e_ps
            Rex[pos] = fit.Rex
            noe_pred_rigid[pos] = fit.NOE_pred_rigid
            noe_flag[pos] = fit.noe_flag
            r2_flag[pos] = fit.r2_flag

        t["S2"] = S2
        t["S2_err"] = S2_err
        t["mf_model"] = mf_model
        t["tau_e_ps"] = tau_e_ps
        t["Rex"] = Rex
        t["NOE_pred_rigid"] = noe_pred_rigid
        t["noe_flag"] = noe_flag
        t["r2_flag"] = r2_flag
        self.table = t

        counts = pd.Series([m for m in mf_model if m is not None]).value_counts().to_dict()
        print(f"  order-parameter fit at {field_mhz:.1f} MHz "
              f"(tau_scale={self.tau_scale:.3f}, {len(calibration)} "
              f"calibration residues): {sum(counts.values())} residues fit {counts}")
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
                if row.get("has_HN", False):
                    labels.append(NODATA)
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
            gray = no data
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
            elif lab == NODATA:
                ax.axvline(j, color="tab:grey", alpha=0.5, lw=0.5)
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