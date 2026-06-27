import numpy as np
import pandas as pd


def compute_r2eff(fit_results, vcpmg_values, time_T2, duplicate_errors=True):
    """
    Compute effective R₂ from fitted peak volumes.

    R2eff(νCPMG) = -(1/T) × ln(I(νCPMG) / I₀)

    Noise errors are propagated from per-plane fit uncertainties. When
    duplicate_errors=True (default), any duplicated νCPMG points are also used
    to estimate a max-absolute-deviation error floor; whichever is larger
    (noise or duplicate-based) is kept in R2eff_err, and the source is recorded
    in the `larger_err` column ('noise' or 'duplicates').

    Parameters
    ----------
    fit_results : DataFrame
        Output of fit_peaks.
    vcpmg_values : list of float
        νCPMG frequencies in Hz, one per CPMG plane (not including the reference).
    time_T2 : float
        Constant-time CPMG delay in seconds.
    duplicate_errors : bool
        If True (default), apply duplicate-based error estimation on top of
        noise propagation.

    Returns
    -------
    DataFrame with columns: ref_index, vcpmg, t_cpmg, R2eff, R2eff_err,
    vol, vol_err, vol_ref, vol_ref_err, N_ppm, H_ppm,
    and (when duplicate_errors=True) R2eff_err_dup, larger_err.
    """
    ref_vols = fit_results[fit_results.plane == 0].set_index("ref_index")["vol"].to_dict()
    ref_errs = fit_results[fit_results.plane == 0].set_index("ref_index")["vol_err"].to_dict()

    rows = []
    for plane_idx, vc in enumerate(vcpmg_values, start=1):
        sub = fit_results[fit_results.plane == plane_idx].copy()
        sub["vol_ref"] = sub["ref_index"].map(ref_vols)
        sub["vol_ref_err"] = sub["ref_index"].map(ref_errs)
        sub["vcpmg"] = vc
        sub["t_cpmg"] = time_T2
        valid = (sub.vol > 0) & (sub.vol_ref > 0)
        sub["R2eff"] = np.nan
        sub["R2eff_err"] = np.nan
        I, I0 = sub.loc[valid, "vol"], sub.loc[valid, "vol_ref"]
        sI, sI0 = sub.loc[valid, "vol_err"], sub.loc[valid, "vol_ref_err"]
        sub.loc[valid, "R2eff"] = -1 / time_T2 * np.log(I / I0)
        sub.loc[valid, "R2eff_err"] = (1 / time_T2) * np.sqrt((sI / I)**2 + (sI0 / I0)**2)
        rows.append(sub[["ref_index", "vcpmg", "t_cpmg", "R2eff", "R2eff_err",
                          "vol", "vol_err", "vol_ref", "vol_ref_err", "N_ppm", "H_ppm"]])

    out = pd.concat(rows, ignore_index=True)
    if duplicate_errors:
        out = _apply_duplicate_errors(out)
    return out


def _apply_duplicate_errors(all_r2eff):
    """
    Replace noise-propagated errors with duplicate-based errors when larger.

    For each peak (ref_index):
      1. Find νCPMG values that appear more than once.
      2. Compute std(R2eff) for each duplicate group.
      3. Max std across all groups → R2eff_err_dup, applied
         uniformly to every νCPMG point for this peak.
      4. Use whichever is larger (dup or noise); record source in `larger_err`.

    Adds columns R2eff_err_dup and larger_err; updates R2eff_err in place.
    """
    df = all_r2eff.copy()
    df["R2eff_err_dup"] = np.nan
    df["larger_err"] = "noise"

    found_any = False

    for ref_idx, grp in df.groupby("ref_index"):
        idx = grp.index
        dup_vcpmg = grp["vcpmg"].value_counts()
        dup_vcpmg = dup_vcpmg[dup_vcpmg > 1].index

        if len(dup_vcpmg) == 0:
            continue

        found_any = True
        dup_rows = grp[grp["vcpmg"].isin(dup_vcpmg)].dropna(subset=["R2eff"])
        if len(dup_rows) == 0:
            continue

        max_dev = dup_rows.groupby("vcpmg")["R2eff"].std().max()
        df.loc[idx, "R2eff_err_dup"] = max_dev

        mean_noise = grp["R2eff_err"].dropna().mean()
        if max_dev > mean_noise:
            df.loc[idx, "larger_err"] = "duplicates"
            df.loc[idx, "R2eff_err"] = max_dev

    if not found_any:
        print("  No duplicate νCPMG points — using noise-propagated errors.")

    return df
