import re
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from ..utils.constants import _AA_3TO1
from ..hydronmr import run as run_hydronmr
from .config import _yaml_set_cache


def _parse_seqpos(assn_label):
    if assn_label is None or (isinstance(assn_label, float) and np.isnan(assn_label)):
        return np.nan
    m = re.search(r"(\d+)", str(assn_label))
    return int(m.group(1)) if m else np.nan


def validate_sequence(sequence, bmrb_peaks):
    """
    Check that a construct sequence agrees with BMRB-derived assignments.

    For each row in bmrb_peaks, sequence[Seq_ID - 1] must equal the
    one-letter code for Comp_ID (e.g. 'P' for PRO). Seq_ID (BMRB's
    1-indexed sequence position) is used rather than Auth_seq_ID (the
    author/PDB residue number), since the latter often has an arbitrary
    offset relative to a 1-indexed construct sequence.

    Parameters
    ----------
    sequence : str
        1-indexed sequence string (sequence[0] is residue 1).
    bmrb_peaks : DataFrame
        Output of fetch_bmrb_peaks; must contain Seq_ID and Comp_ID.

    Returns
    -------
    list of dict
        One entry per mismatch, with keys: seqpos, expected (from sequence),
        observed (from BMRB), assn_label. Empty if everything matches.

    Raises
    ------
    ValueError
        If any Seq_ID falls outside the sequence (1..len(sequence)).
    """
    mismatches = []
    for _, row in bmrb_peaks.iterrows():
        seqpos = int(row["Seq_ID"])
        if seqpos < 1 or seqpos > len(sequence):
            raise ValueError(
                f"BMRB residue {row['assn_label']} (Seq_ID={seqpos}) "
                f"is outside sequence (length {len(sequence)})"
            )
        expected = sequence[seqpos - 1]
        observed = _AA_3TO1.get(row["Comp_ID"], "?")
        if expected != observed:
            mismatches.append(dict(seqpos=seqpos, expected=expected,
                                   observed=observed, assn_label=row["assn_label"]))

    if mismatches:
        print(f"  WARNING: {len(mismatches)} sequence/BMRB mismatch(es):")
        for m in mismatches:
            print(f"    position {m['seqpos']}: sequence has '{m['expected']}', "
                  f"BMRB ({m['assn_label']}) has '{m['observed']}'")
    else:
        print(f"  Sequence matches BMRB assignments ({len(bmrb_peaks)} residues checked).")

    return mismatches


def flatten_r2eff(all_r2eff, ref_df, max_r2err_threshold=100.0, start_num=2, end_num=3):
    """
    Reduce all_r2eff to one row per residue with R2first, R2last, and spread stats.
    Only well-fit peaks (>=4 points, all errors below threshold) are kept.
    Returns DataFrame with ref_index, seqpos, assn_label, R2first, R2last,
    std_first, std_last.
    """
    rows = []
    for ref_idx, grp in all_r2eff.groupby("ref_index"):
        grp = grp.dropna(subset=["R2eff", "R2eff_err"]).sort_values("vcpmg")
        if len(grp) < 4 or grp["R2eff_err"].max() >= max_r2err_threshold:
            continue
        firstfew = grp.iloc[:start_num]["R2eff"]
        lastfew  = grp.iloc[-end_num:]["R2eff"]

        rows.append(dict(
            ref_index=ref_idx,
            R2first=firstfew.mean(),
            R2last=lastfew.mean(),
            std_first=grp.iloc[:start_num]["R2eff_err"].mean(),
            std_last=lastfew.max() - lastfew.min(),
            std_total=grp["R2eff"].max() - grp["R2eff"].min(),
        ))
    df = pd.DataFrame(rows)
    df = df.merge(ref_df[["ref_index", "assn_label"]], on="ref_index", how="left")
    df["seqpos"] = df["assn_label"].apply(_parse_seqpos)
    return df


def fit_R2_rigid(df, hydro_df):
    """
    Merge HYDRONMR R2 predictions onto df (by seqpos), fit a scale factor
    C so that C*R2_hydro ≈ R2last for the most rigid residues, and return
    df with scaled_R2_pred added.
    """
    df = df.merge(hydro_df[["seqpos", "R2_hydro"]], on="seqpos", how="left")
    ma = (df["std_last"] < np.nanmedian(df["std_last"])) & df["R2_hydro"].notna()
    scalar = float((df.loc[ma, "R2last"] * df.loc[ma, "R2_hydro"]).sum()
                   / (df.loc[ma, "R2_hydro"] ** 2).sum())
    df["scaled_R2_pred"] = scalar * df["R2_hydro"]
    residuals = df.loc[ma, "R2last"] - df.loc[ma, "scaled_R2_pred"]
    df["rigid_rmse"] = float(np.sqrt(np.mean(residuals**2)))
    print(f"  HYDRONMR scale factor: {scalar:.3f}  ({ma.sum()} rigid residues used, RMSE={df['rigid_rmse'].iloc[0]:.3f})")
    return df


def _calc_rigid_R2(config, overwrite_cache=False):
    """
    Run HYDRONMR on the PDB structure in config["pdb"] and return per-residue
    predicted R2 values.

    Caching: checks for an existing cache in this order:
      1. hydronmr_r2_cache key in the YAML file
      2. <yaml_stem>_hydronmr_r2.csv next to the YAML

    Parameters
    ----------
    config : dict
        Output of load_config. Must contain a "pdb" key.
    overwrite_cache : bool
        If True, re-run HYDRONMR even if a cached CSV exists.

    Returns
    -------
    DataFrame with columns: seqpos (int), R2_hydro (float)
    """
    if "pdb" not in config:
        raise ValueError("config is missing 'pdb' (path to PDB structure for HYDRONMR)")

    yaml_path = config["yaml_path"]
    cache_dir = yaml_path.parent
    sequence = config.get("sequence")
    n_residues = len(sequence) if sequence else None

    if not overwrite_cache:
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        if "hydronmr_r2_cache" in cfg:
            configured = Path(cfg["hydronmr_r2_cache"])
            if configured.exists():
                df = pd.read_csv(configured, usecols=lambda c: c in ("seqpos", "R2_hydro"))
                print(f"  {len(df)} rows loaded from HYDRONMR cache → {configured.name}")
                if n_residues is not None:
                    df = df[df["seqpos"] <= n_residues].reset_index(drop=True)
                df.attrs["cache_path"] = configured
                return df

        default_cache = cache_dir / f"{yaml_path.stem}_hydronmr_r2.csv"
        if default_cache.exists():
            df = pd.read_csv(default_cache, usecols=lambda c: c in ("seqpos", "R2_hydro"))
            print(f"  {len(df)} rows loaded from HYDRONMR cache → {default_cache.name}")
            _yaml_set_cache(yaml_path, default_cache, key="hydronmr_r2_cache")
            if n_residues is not None:
                df = df[df["seqpos"] <= n_residues].reset_index(drop=True)
            df.attrs["cache_path"] = default_cache
            return df

    pdb_path = Path(config["pdb"])
    result = run_hydronmr(pdb_path)
    rows = [dict(seqpos=resseq, R2_hydro=1.0 / t2)
            for (chain, resseq), (t1, t2, ratio, noe) in result.per_residue.items()]
    df = pd.DataFrame(rows)
    # Multi-chain structures (e.g. a crystallographic dimer) can have
    # multiple chains sharing the same residue numbering; average their
    # predictions so seqpos is unique.
    df = df.groupby("seqpos", as_index=False)["R2_hydro"].mean()
    df = df.sort_values("seqpos").reset_index(drop=True)

    cache_path = cache_dir / f"{yaml_path.stem}_hydronmr_r2.csv"
    df.to_csv(cache_path, index=False)
    print(f"  Saved HYDRONMR R2 cache → {cache_path.name}")
    _yaml_set_cache(yaml_path, cache_path, key="hydronmr_r2_cache")

    if n_residues is not None:
        df = df[df["seqpos"] <= n_residues].reset_index(drop=True)
    df.attrs["cache_path"] = cache_path
    return df


def classify_peaks(all_r2eff, ref_df, config, max_r2err_threshold=100.0):
    """Classify each peak as 'Rex', 'elevated_R2', or 'flat'."""
    start_num = config["start_num"]
    end_num = config["end_num"]
    df = flatten_r2eff(all_r2eff, ref_df, max_r2err_threshold, start_num, end_num)
    hydro_df = _calc_rigid_R2(config)
    df = fit_R2_rigid(df, hydro_df)

    base_uncertainty = float(df["std_total"].median())
    print(f"  base_uncertainty: {base_uncertainty:.3f} s⁻¹ (median std_total)")

    df["Rex_val"] = df["R2first"] - df["R2last"]
    df["Rex_err"] = df["std_first"] + df["std_last"]
    df["Rex"] = ((df["Rex_val"] - df["Rex_err"]) > base_uncertainty)
    df["label"] = np.where(df["Rex"], "Rex", "flat")

    rigid_rmse = float(df["rigid_rmse"].iloc[0])
    df["elevated_R2"] = (df["scaled_R2_pred"].notna()
        & (df["R2last"] - df["std_last"] > df["scaled_R2_pred"] + rigid_rmse + base_uncertainty))
    df["label"] = np.where(df["elevated_R2"], "elevated_R2", df["label"])

    return df.sort_values("seqpos").reset_index(drop=True)
