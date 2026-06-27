import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .spectrum import estimate_background


def plot_spectrum(ref_data, contour_levels=8, cmap="plasma",
                  contour_color=None, linewidth=0.8, vmax_scale=0.5,
                  xlim=None, ylim=None, baseline=10,
                  figsize=(8, 7), ax=None):
    """
    Plot 2D spectrum contours.

    Use `plot_peaklist` to overlay peak markers/labels on the returned axes 
    """
    uc = ref_data.uc
    data = ref_data.data
    ppm_w1 = uc[0].ppm_scale()
    ppm_w2 = uc[1].ppm_scale()
    vmax   = data.max() * vmax_scale
    noise  = estimate_background(data)
    levels = np.logspace(np.log10(noise * baseline), np.log10(vmax),
                         contour_levels)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if contour_color is not None:
        ax.contour(ppm_w2, ppm_w1, data, levels,
                   colors=contour_color, linewidths=linewidth, alpha=0.6)
    else:
        ax.contour(ppm_w2, ppm_w1, data, levels, cmap=cmap, linewidths=linewidth)

    ax.set_xlabel(r"$^1$H (ppm)", fontsize=14)
    ax.set_ylabel(r"$^{15}$N (ppm)", fontsize=14)

    # fix if axis labels are ordered small->large
    if xlim is not None and xlim[0]<xlim[1]:
        xlim=(xlim[1],xlim[0])
    if ylim is not None and ylim[0]<ylim[1]:
        ylim=(ylim[1],ylim[0])

    ax.set_xlim(xlim if xlim else (ppm_w2.max(), ppm_w2.min()))
    ax.set_ylim(ylim if ylim else (ppm_w1.max(), ppm_w1.min()))
    return fig, ax


def plot_peaklist(ax=None, peaks_df=None, marker="x", peaks_xcol="H_ppm", peaks_ycol="N_ppm",
                  color="limegreen", markersize=3,
                  text="ref_index", label_fontsize=6,
                  label=None,
                  hue=None, palette=None,
                  figsize=(8, 6)):
    """
    Plot peak markers (and optional labels), optionally onto an existing axes.

    Parameters
    ----------
    ax : matplotlib Axes or None
        Axes to plot onto. If None, a new figure is created with inverted axes
        (H on x, N on y, both increasing toward origin as in NMR convention).
    peaks_df : DataFrame
        Peaks to plot.
    markersize : float
        Marker size for peak positions.
    text : str or None
        Column in peaks_df to use for per-peak text annotations. Pass None
        to suppress these.
    label : str or None
        Legend label for this peaklist. When hue is None, passed straight to
        ax.plot (and a legend is drawn if set). When hue is set, this is
        ignored — each hue group gets its own legend entry instead.
    hue : str or None
        Column in peaks_df to colour peaks by. When set, each unique value gets
        its own colour from `palette`, markers are grouped in the legend, and
        annotation text is coloured to match.  Inspired by seaborn's hue/palette API.
    palette : dict or None
        Maps hue values to matplotlib colours. Values absent from the dict
        fall back to 'gray'. Ignored when hue is None.
    """
    def _label_text(val):
        try:
            return str(int(val))
        except (ValueError, TypeError):
            return str(val)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlabel("$^1$H (ppm)")
        ax.set_ylabel("$^{15}$N (ppm)")
    else:
        fig = ax.get_figure()

    if peaks_df is None or peaks_xcol not in peaks_df.columns or peaks_ycol not in peaks_df.columns:
        return (fig, ax) if standalone else ax

    if not standalone:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_lo, x_hi = min(xlim), max(xlim)
        y_lo, y_hi = min(ylim), max(ylim)
        in_view = (peaks_df[peaks_xcol].between(x_lo, x_hi)
                   & peaks_df[peaks_ycol].between(y_lo, y_hi))
        peaks_df = peaks_df[in_view]

    if hue is not None and hue in peaks_df.columns:
        if palette is None:
            default_colors = sns.color_palette("tab10")
            pal = {val: default_colors[i % len(default_colors)]
                   for i, val in enumerate(peaks_df[hue].unique())}
        else:
            pal = palette
        for group_val, grp in peaks_df.groupby(hue):
            color = pal.get(group_val, "gray")
            ax.plot(grp[peaks_xcol], grp[peaks_ycol],
                    marker, color=color, ms=markersize, mew=1.2, zorder=5,
                    label=f"{hue}: {group_val}")
            if text is not None and text in grp.columns:
                for _, row in grp.iterrows():
                    ax.text(row[peaks_xcol] - 0.02, row[peaks_ycol] - 0.2,
                            _label_text(row[text]), color=color,
                            fontsize=label_fontsize, zorder=6,
                            ha="left", va="bottom")
        ax.legend(loc="upper left", fontsize=11)
    else:
        ax.plot(peaks_df[peaks_xcol], peaks_df[peaks_ycol],
                marker, color=color, ms=markersize, mew=1.2, zorder=5,
                label=label)
        if text is not None and text in peaks_df.columns:
            for _, row in peaks_df.iterrows():
                ax.text(row[peaks_xcol] - 0.02, row[peaks_ycol] - 0.2,
                        _label_text(row[text]), color=color,
                        fontsize=label_fontsize, zorder=6,
                        ha="left", va="bottom")
        if label is not None:
            ax.legend(loc="upper left", fontsize=11)

    return (fig, ax) if standalone else ax


def plot_csp(peaks_df1, peaks_df2, on,
             xcol="H_ppm", ycol="N_ppm",
             color1="steelblue", color2="tab:orange",
             line_color="gray", line_alpha=0.5,
             marker="o", markersize=4,
             text=None, label_fontsize=6,
             ax=None, figsize=(8, 6)):
    """
    Plot two matched peaklists and draw connecting lines between paired peaks.

    Parameters
    ----------
    peaks_df1, peaks_df2 : DataFrame
        The two peaklists to compare (e.g. apo and holo, or two conditions).
    on : str or list of str
        Column(s) to merge on — the shared identifier between the two
        peaklists (e.g. 'assn_label', 'Seq_ID', or ['chain', 'Seq_ID']).
    xcol, ycol : str
        Column names for H and N chemical shifts in both DataFrames.
    color1, color2 : str
        Marker colours for peaks_df1 and peaks_df2 respectively.
    line_color, line_alpha : str, float
        Style for the connecting lines.
    marker : str
        Matplotlib marker string applied to both peaklists.
    markersize : float
    text : str or None
        Column in peaks_df1 to use for per-peak annotations. Pass None to suppress.
    label_fontsize : int
    ax : matplotlib Axes or None
        Axes to plot onto. Creates a new figure if None.
    figsize : tuple

    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlabel("$^1$H (ppm)")
        ax.set_ylabel("$^{15}$N (ppm)")
    else:
        fig = ax.get_figure()

    merged = peaks_df1.merge(peaks_df2, on=on, suffixes=("_1", "_2"))
    xcol1 = xcol + "_1" if xcol + "_1" in merged.columns else xcol
    ycol1 = ycol + "_1" if ycol + "_1" in merged.columns else ycol
    xcol2 = xcol + "_2" if xcol + "_2" in merged.columns else xcol
    ycol2 = ycol + "_2" if ycol + "_2" in merged.columns else ycol

    for _, row in merged.iterrows():
        ax.plot([row[xcol1], row[xcol2]], [row[ycol1], row[ycol2]],
                color=line_color, alpha=line_alpha, linewidth=0.8, zorder=2)

    ax.plot(merged[xcol1], merged[ycol1],
            marker, color=color1, ms=markersize, mew=1.2, zorder=3, label="peaks_df1")
    ax.plot(merged[xcol2], merged[ycol2],
            marker, color=color2, ms=markersize, mew=1.2, zorder=3, label="peaks_df2")

    if text is not None and text in merged.columns:
        for _, row in merged.iterrows():
            ax.text(row[xcol1] - 0.02, row[ycol1] - 0.2,
                    str(row[text]), color=color1,
                    fontsize=label_fontsize, zorder=4,
                    ha="left", va="bottom")

    return fig, ax