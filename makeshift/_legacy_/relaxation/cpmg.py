import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from .config import load_config, load_planes
from .lineshape import fit_peaks
from .r2eff import compute_r2eff
from .classify import classify_peaks, validate_sequence, _parse_seqpos
from .plotting import plot_r2eff_per_peak, plot_r2eff_grid, plot_waterfall
from ..spectrum.peaks import pick_peaks, load_peaklist
from ..spectrum.matching import map_peaklists
from ..plotting.plotting import plot_spectrum, plot_peaklist


def save_results_csv(ref_df, classifications, csv_path, fit_results=None, missing_df=None):
    """
    Write a per-peak summary CSV combining peak assignments and classification.

    Columns: ref_index, assn_label (if present), N_ppm, H_ppm, vol, vol_err,
    lw_n, lw_h (if fit_results given, from the reference plane),
    R2first, R2last, std_first, std_last, std_total, Rex_val, Rex_err, Rex,
    elevated_R2, label, seqpos, R2_hydro, scaled_R2_pred, rigid_rmse, missing_assignment.

    Rows are sorted by seqpos (peaks without a seqpos are placed last).

    Parameters
    ----------
    ref_df : DataFrame
        Picked peaks, optionally with assn_label.
    classifications : DataFrame
        Output of classify_peaks.
    csv_path : str or Path
    fit_results : DataFrame, optional
        Output of fit_peaks. If given, the reference-plane (plane 0) vol,
        vol_err, lw_n, lw_h are merged in.
    missing_df : DataFrame, optional
        Columns: seqpos, label ("proline" or "no_peaklist_assignment"), as
        computed in run_protein. Appended as extra rows (with NaN peak
        columns) and carried forward in the new "missing_assignment" column.

    Returns
    -------
    DataFrame written to csv_path.
    """
    classifications = classifications.drop(columns=["assn_label"], errors="ignore")
    ref_cols = [c for c in ["ref_index", "assn_label", "N_ppm", "H_ppm"] if c in ref_df.columns]
    out = ref_df[ref_cols].merge(classifications, on="ref_index", how="left")

    if fit_results is not None:
        ref_fit = fit_results[fit_results["plane"] == 0][
            ["ref_index", "vol", "vol_err", "lw_n", "lw_h"]
        ]
        out = out.merge(ref_fit, on="ref_index", how="left")

    out["missing_assignment"] = np.nan

    if missing_df is not None and len(missing_df):
        existing_seqpos = set(out["seqpos"].dropna().astype(int)) if "seqpos" in out.columns else set()
        extra = missing_df[~missing_df["seqpos"].isin(existing_seqpos)].copy()
        extra = extra.rename(columns={"label": "missing_assignment"})
        out = pd.concat([out, extra], ignore_index=True)

    if "seqpos" in out.columns:
        out = out.sort_values("seqpos", na_position="last").reset_index(drop=True)

    out.to_csv(csv_path, index=False)
    print(f"  Saved results CSV → {Path(csv_path).name}")
    return out


def run_protein(yaml_path, out_dir, lineshape="gaussian",
                max_r2err_threshold=10000.0, color_map=None,
                xlim=(10.5, 6.0), ylim=(135, 100),
                zoom_xlim=(9, 7), zoom_ylim=(125, 112)):
    """
    Run the full CPMG pipeline for one protein and write standard outputs.

    Steps: load config/planes -> pick peaks -> (optional) BMRB assignment ->
    fit lineshapes -> compute R2eff -> classify peaks (with HYDRONMR
    elevated_R2 classification) -> write plots and CSV.

    Parameters
    ----------
    yaml_path : str or Path
        CPMG config YAML (see load_config).
    out_dir : str or Path
        Output directory; created if absent. Files are named
        <yaml_stem>_<thing>.<ext>.
    lineshape, max_r2err_threshold :
        Passed to fit_peaks / classify_peaks. Peak-picking baseline,
        matching tolerance, start_num, and end_num are read from config
        (see load_config for defaults).
    color_map : dict, optional
        Classification -> colour, used for the classified HSQC and waterfall
        plots. Defaults to a built-in CPMG_COLORS-style map.
    xlim, ylim : tuple
        ppm window for HSQC plots.
    zoom_xlim, zoom_ylim : tuple
        ppm window for the zoomed-in panel of the peak-mapping plot.

    Returns
    -------
    dict with keys: ref_df, fit_results, all_r2eff, classifications.
    """
    if color_map is None:
        color_map = {
            "Rex": "tab:orange",
            "elevated_R2": "#8ACE00",
            "flat": "black",
            "unfit": "red",
        }

    yaml_path = Path(yaml_path)
    out_dir = Path(out_dir) / "outputs"
    cache_dir = Path(out_dir).parent / "caches"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = yaml_path.stem

    print("== Loading config and planes ==")
    config = load_config(yaml_path)
    ref_spectrum, plane_data, vcpmg_values, time_T2 = load_planes(config)

    print("== Picking peaks ==")
    ref_df = pick_peaks(ref_spectrum, baseline=config["baseline_ref_plane"])
    ref_df["ref_index"] = ref_df.index
    print(f"  {len(ref_df)} peaks picked")

    fig, ax = plot_spectrum(ref_spectrum, baseline=config["baseline_ref_plane"], xlim=xlim, ylim=ylim, cmap="Greys_r")
    ax.set_title(f"{stem} — reference HSQC")
    fig.savefig(out_dir / f"{stem}_hsqc_unlabeled.pdf", bbox_inches="tight")
    plt.close(fig)

    print("== Peak assignment ==")
    sequence = config.get("sequence")
    peaklist = config.get("peaklist")
    missing_df = None
    if peaklist and sequence:
        peaks_for_missing = load_peaklist(peaklist)
        assigned_seqpos = set(
            peaks_for_missing["assn_label"].apply(_parse_seqpos).dropna().astype(int)
        )
        rows = []
        for i, char in enumerate(sequence):
            seqpos = i + 1
            if char == "P":
                rows.append(dict(seqpos=seqpos, label="proline"))
            elif seqpos not in assigned_seqpos:
                rows.append(dict(seqpos=seqpos, label="no_peaklist_assignment"))
        missing_df = pd.DataFrame(rows)
    if peaklist:
        picked_df = ref_df.copy()
        peaks = load_peaklist(peaklist)
        if sequence:
            validate_sequence(sequence, peaks)

        ref_df, peaks_shifted = map_peaklists(ref_df, peaks, tol=config["peak_mapping_tol"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, panel_xlim, panel_ylim in zip(axes, (xlim, zoom_xlim), (ylim, zoom_ylim)):
            plot_spectrum(ref_spectrum, baseline=config["baseline_ref_plane"],
                          xlim=panel_xlim, ylim=panel_ylim, cmap="Greys_r", ax=ax)
            plot_peaklist(ax, picked_df, text=None, color="tab:blue", label="picked peaks")
            plot_peaklist(ax, peaks_shifted, text="assn_label", color="tab:green", label="ref peaklist, translated")
            plot_peaklist(ax, ref_df, text="assn_label", color="magenta", label="mapped peaks")
        fig.suptitle(f"{stem} — peak mapping")
        plt.tight_layout()
        fig.savefig(out_dir / f"{stem}_peak_mapping.pdf", bbox_inches="tight")
        plt.close(fig)
    else:
        print("  No peaklist in config — skipping.")

    print("== Fitting lineshapes ==")
    fit_results = fit_peaks(plane_data, ref_df, ref_spectrum.uc, config, lineshape=lineshape, cache_dir=cache_dir)

    print("== Computing R2eff ==")
    all_r2eff = compute_r2eff(fit_results, vcpmg_values, time_T2)

    print("== Classifying peaks ==")
    classifications = classify_peaks(all_r2eff, ref_df, config,
                                     max_r2err_threshold=max_r2err_threshold)
    print(classifications["label"].value_counts().to_string())

    merged = ref_df.merge(classifications[["ref_index", "label"]], on="ref_index", how="left")
    merged["label"] = merged["label"].fillna("unlabeled")

    text_col = ("assn_label"
                if "assn_label" in merged.columns and merged["assn_label"].notna().any()
                else "ref_index")

    fig, ax = plot_spectrum(ref_spectrum, baseline=config["baseline_ref_plane"],
                            contour_color="black", xlim=xlim, ylim=ylim,
                            figsize=(9, 8))
    plot_peaklist(ax, merged, marker=".", markersize=4,
                  text=text_col, label_fontsize=6,
                  hue="label", palette=color_map)
    ax.set_title(f"{stem} — CPMG classification", fontsize=14)
    fig.savefig(out_dir / f"{stem}_hsqc_annotated_assigned.pdf", bbox_inches="tight")
    plt.close(fig)

    print("== Plotting dispersion grid ==")
    fig, _ = plot_r2eff_grid(all_r2eff, classifications, ref_df=ref_df, color_map=color_map)
    fig.savefig(out_dir / f"{stem}_grid.pdf", bbox_inches="tight")
    plt.close(fig)

    print("== Plotting waterfall ==")
    fig, ax = plot_waterfall(all_r2eff, classifications, ref_df, color_map=color_map,
                             sequence=sequence, missing_df=missing_df)
    fig.savefig(out_dir / f"{stem}_waterfall.pdf", bbox_inches="tight")
    plt.close(fig)

    print("== Writing CSV ==")
    save_results_csv(ref_df, classifications, out_dir / f"{stem}.csv",
                     fit_results=fit_results, missing_df=missing_df)

    return dict(ref_df=ref_df, fit_results=fit_results, all_r2eff=all_r2eff,
                classifications=classifications)
