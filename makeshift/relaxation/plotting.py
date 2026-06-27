import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


def plot_r2eff_per_peak(all_r2eff, classifications, pdf_path, ref_df=None):
    """
    Write a multi-page PDF with one page per peak showing R2eff vs CPMG.

    Parameters
    ----------
    all_r2eff : DataFrame
        Output of compute_r2eff.
    classifications : DataFrame
        Output of classify_peaks.
    pdf_path : str or Path
    ref_df : DataFrame, optional
        If provided and contains a 'assn_label' column, the BMRB label is
        prepended to each page title.
    """
    class_map = classifications.set_index("ref_index")
    has_dup_col = "larger_err" in all_r2eff.columns and "R2eff_err_dup" in all_r2eff.columns

    bmrb_map = {}
    if ref_df is not None and "assn_label" in ref_df.columns:
        bmrb_map = ref_df.set_index("ref_index")["assn_label"].dropna().to_dict()

    with PdfPages(pdf_path) as pdf:
        for ref_idx, grp in all_r2eff.groupby("ref_index"):
            grp = grp.dropna(subset=["R2eff"]).sort_values("vcpmg")
            if len(grp) == 0 or ref_idx not in class_map.index:
                continue

            cls = class_map.loc[ref_idx]

            larger_err = grp["larger_err"].iloc[0] if has_dup_col else "noise"
            if larger_err == "duplicates":
                yerr = grp["R2eff_err_dup"].fillna(grp["R2eff_err"]).values
            else:
                yerr = grp["R2eff_err"].values

            N_ppm = grp["N_ppm"].iloc[0]
            H_ppm = grp["H_ppm"].iloc[0]
            bmrb_lbl = bmrb_map.get(ref_idx)
            prefix = f"{bmrb_lbl}  " if bmrb_lbl else ""

            fig, ax = plt.subplots(figsize=(4, 3))
            ax.errorbar(grp["vcpmg"].values, grp["R2eff"].values, yerr=yerr,
                        capsize=2, fmt=".", color="tab:blue")

            if cls is not None and not np.isnan(cls["R2last"]):
                ax.axhline(cls["R2last"], linestyle=":", zorder=0, color="grey")
                x0, x1 = ax.get_xlim()
                ax.fill_between(
                    [x0, x1],
                    cls["R2last"] - cls["std_last"],
                    cls["R2last"] + cls["std_last"],
                    alpha=0.1, zorder=0, color="grey", linewidth=0,
                )
                ax.set_xlim([x0, x1])

            ax.set_xlabel(r"$\nu_{CPMG}$ (Hz)")
            ax.set_ylabel(r"$R_{2,\mathrm{eff}}$ (s$^{-1}$)")

            if cls is not None:
                title = (
                    f"{prefix}peak {ref_idx}  ({N_ppm:.1f}, {H_ppm:.2f} ppm)\n"
                    f"Rex = {cls['Rex_val']:.2f} ± {cls['Rex_err']:.2f} s⁻¹"
                    f"  |  {cls['label']}  |  err: {larger_err}"
                )
            else:
                title = f"{prefix}peak {ref_idx}  ({N_ppm:.1f}, {H_ppm:.2f} ppm)"
            ax.set_title(title, fontsize=7)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"  Saved per-peak PDF → {Path(pdf_path).name}")


def plot_r2eff_grid(all_r2eff, classifications, ref_df=None,
                    color_map=None, ncols=10,
                    figsize_per_panel=(2.5, 2.0),
                    title_fontsize=6, axis_fontsize=7):
    """
    Plot all R2eff dispersion curves in a compact grid, one panel per peak.

    Peaks are sorted N→C by residue number when BMRB labels are available
    (parsed from ref_df['assn_label']).  Unassigned peaks appear at the end.

    Parameters
    ----------
    all_r2eff : DataFrame
        Output of compute_r2eff.
    classifications : DataFrame
        Output of classify_peaks.
    ref_df : DataFrame, optional
        If provided and contains a 'assn_label' column, labels are used as
        panel titles and peaks are sorted N→C.
    color_map : dict, optional
        Maps classification labels to matplotlib colours. Missing keys → 'black'.
    ncols : int
        Number of columns in the grid (default 10).
    figsize_per_panel : tuple
        (width, height) in inches per panel.
    title_fontsize, axis_fontsize : int
        Font sizes for panel titles and tick labels.

    Returns
    -------
    fig, axes
    """
    if color_map is None:
        color_map = {
            "Rex": "tab:orange",
            "elevated_R2": "#8ACE00",
            "flat": "black",
            "unfit": "red",
        }

    class_map = classifications.set_index("ref_index")

    has_bmrb = (ref_df is not None
                and "assn_label" in ref_df.columns
                and ref_df["assn_label"].notna().any())

    ref_indices = list(all_r2eff["ref_index"].unique())

    if has_bmrb:
        bmrb_map = ref_df.set_index("ref_index")["assn_label"].to_dict()

        def _res_num(idx):
            lbl = bmrb_map.get(idx)
            if lbl is None or (isinstance(lbl, float) and np.isnan(lbl)):
                return 99999
            m = re.search(r"(\d+)", str(lbl))
            return int(m.group(1)) if m else 99999

        ref_indices = sorted(ref_indices, key=_res_num)
    else:
        bmrb_map = {}
        ref_indices = sorted(ref_indices)

    n_peaks = len(ref_indices)
    nrows   = int(np.ceil(n_peaks / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * figsize_per_panel[0], nrows * figsize_per_panel[1]),
        squeeze=False,
    )

    for i, ref_idx in enumerate(ref_indices):
        row, col = divmod(i, ncols)
        ax = axes[row][col]

        grp = (all_r2eff[all_r2eff["ref_index"] == ref_idx]
               .dropna(subset=["R2eff"])
               .sort_values("vcpmg"))

        cls   = class_map.loc[ref_idx] if ref_idx in class_map.index else None
        label = cls["label"] if cls is not None else "unfit"
        color = color_map.get(label, "black")

        if len(grp) > 0:
            ax.errorbar(grp["vcpmg"], grp["R2eff"], yerr=grp["R2eff_err"],
                        fmt=".", color=color, capsize=1.5, ms=3, lw=0.8, elinewidth=0.8)

            if cls is not None and not np.isnan(cls["R2last"]):
                ax.axhline(cls["R2last"], linestyle=":", lw=0.8, color="grey", zorder=0)
                x0, x1 = ax.get_xlim()
                ax.fill_between([x0, x1],
                                cls["R2last"] - cls["std_last"],
                                cls["R2last"] + cls["std_last"],
                                alpha=0.15, color="grey", linewidth=0, zorder=0)
                ax.set_xlim([x0, x1])

            if (cls is not None and "scaled_R2_pred" in cls.index
                    and not pd.isna(cls["scaled_R2_pred"])):
                x0, x1 = ax.get_xlim()
                ax.axhline(cls["scaled_R2_pred"], linestyle="-", lw=0.8,
                          color="k", zorder=1)
                ax.set_xlim([x0, x1])

        bmrb_lbl = bmrb_map.get(ref_idx)
        if bmrb_lbl and not (isinstance(bmrb_lbl, float) and np.isnan(bmrb_lbl)):
            title = str(bmrb_lbl)
        else:
            title = str(ref_idx)

        if cls is not None and cls["label"] == "Rex" and not np.isnan(cls["Rex_val"]):
            title += f"\n{cls['Rex_val']:.1f}±{cls['Rex_err']:.1f} s⁻¹"

        ax.set_title(title, fontsize=title_fontsize, color="black")
        ax.tick_params(labelsize=title_fontsize - 1)

        ymin, ymax = ax.get_ylim()
        if ymax - ymin < 2:
            mid = (ymin + ymax) / 2
            ax.set_ylim(mid - 1, mid + 1)

    for j in range(n_peaks, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row][col].set_visible(False)

    for r in range(nrows):
        axes[r][0].set_ylabel(r"$R_{2,\mathrm{eff}}$ (s$^{-1}$)", fontsize=axis_fontsize)
    for c in range(ncols):
        axes[-1][c].set_xlabel(r"$\nu_\mathrm{CPMG}$ (Hz)", fontsize=axis_fontsize)

    plt.tight_layout()
    return fig, axes


def plot_waterfall(all_r2eff, classifications, ref_df, color_map=None, sequence=None, missing_df=None):
    """
    Plot R2eff vs sequence position for every CPMG point, with vertical bands 
    marking peak classification, prolines, and missing residues.

    Parameters
    ----------
    all_r2eff : DataFrame
        Output of compute_r2eff.
    classifications : DataFrame
        Output of classify_peaks; scaled_R2_pred is overlaid as a black line.
    ref_df : DataFrame
        Must contain ref_index and assn_label columns (used to map peaks to
        sequence positions).
    color_map : dict, optional
        Maps classification labels to matplotlib colours for the axvline
        bands. Missing keys are not shaded.
    sequence : str, optional
        One-letter sequence string, 1-indexed by position, with 'P' marking
        prolines. When given, the x-axis spans the full sequence and
        prolines/missing residues are annotated.
    missing_df : DataFrame, optional
        Columns: seqpos, label, where label is "proline" or
        "no_peaklist_assignment". Drawn as vertical bands.

    Returns
    -------
    fig, ax
    """
    if color_map is None:
        color_map = {
            "Rex": "tab:orange",
            "elevated_R2": "#8ACE00",
            "no_peaklist_assignment": "tab:red",
        }

    df = classifications.dropna(subset=["seqpos"]).copy()
    df["seqpos"] = df["seqpos"].astype(int)
    seqpos_map = df.set_index("seqpos")
    ref_index_to_seqpos = df.set_index("ref_index")["seqpos"]

    n_res = len(sequence) if sequence is not None else int(df["seqpos"].max())

    fig, ax = plt.subplots(figsize=(0.06 * n_res + 1, 1.5))

    points = all_r2eff.dropna(subset=["R2eff"]).copy()
    points["seqpos"] = points["ref_index"].map(ref_index_to_seqpos)
    points = points.dropna(subset=["seqpos"])
    points["seqpos"] = points["seqpos"].astype(int)

    vcpmg_values = sorted(points["vcpmg"].unique())
    palette = sns.color_palette("rainbow_r", len(vcpmg_values))
    for vc, color in zip(vcpmg_values, palette):
        sub = points[points["vcpmg"] == vc]
        ax.scatter(sub["seqpos"], sub["R2eff"], s=5, color=color, zorder=2)

    pred_line = None
    if "scaled_R2_pred" in df.columns and df["scaled_R2_pred"].notna().any():
        pred_line = df[["seqpos", "scaled_R2_pred", "rigid_rmse"]].dropna(subset=["scaled_R2_pred"]).sort_values("seqpos")

    if pred_line is not None and len(pred_line):
        y = pred_line["scaled_R2_pred"]
        x = pred_line["seqpos"]
        rmse = pred_line["rigid_rmse"].iloc[0]
        ax.plot(x, y, color="k", zorder=3)
        ax.fill_between(x, y - rmse, y + rmse, color="k", alpha=0.15, zorder=2, linewidth=0)

    ax.set_xlim(0, n_res + 1)
    ax.set_xlabel("Residue")
    ax.set_ylabel(r"$R_2$ (s$^{-1}$)")

    ymin, ymax = ax.get_ylim()
    p_pos = ymin + 0.05 * (ymax - ymin)
    star_pos = ymin + 0.9 * (ymax - ymin)

    missing_map = {}
    if missing_df is not None:
        missing_map = missing_df.set_index("seqpos")["label"].to_dict()

    if sequence is not None:
        for i, char in enumerate(sequence):
            seqpos = i + 1
            m_label = missing_map.get(seqpos)
            if m_label == "proline" or char == "P":
                ax.axvline(seqpos, color="tab:purple", zorder=0, alpha=0.5, linewidth=0.5)
                ax.text(seqpos - 0.5, p_pos, "P", color="tab:purple", fontsize=5, weight="bold")
            elif m_label == "no_peaklist_assignment":
                color = color_map.get("no_peaklist_assignment", "tab:red")
                ax.axvline(seqpos, color=color, zorder=0, alpha=0.5, linewidth=0.5)
                ax.scatter([seqpos], [star_pos], marker="*", color=color)
            elif seqpos not in seqpos_map.index:
                ax.axvline(seqpos, color="grey", zorder=0, alpha=0.3, linewidth=0.5)
            else:
                label = seqpos_map.loc[seqpos, "label"]
                if label in ("Rex", "elevated_R2"):
                    color = color_map.get(label)
                    if color is not None:
                        ax.axvline(seqpos, color=color, zorder=0, alpha=0.3)
    else:
        for seqpos, row in seqpos_map.iterrows():
            if row["label"] in ("Rex", "elevated_R2"):
                color = color_map.get(row["label"])
                if color is not None:
                    ax.axvline(seqpos, color=color, zorder=0, alpha=0.3)

    ax.set_ylim(ymin, ymax)
    plt.tight_layout()
    return fig, ax