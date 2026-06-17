import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .io import estimate_background

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_spectrum(ref_data, contour_levels=8, cmap="plasma",
                  contour_color=None, linewidth=0.8, vmax_scale=0.5,
                  xlim=None, ylim=None, baseline=10,
                  figsize=(8, 7), ax=None):
    """
    Plot 2D spectrum contours.

    Parameters
    ----------
    ref_data : Spectrum
        Output of read_ucsf (data + per-axis unit-conversion objects).
    contour_color : str or None
        If set, draw all contours in this solid color instead of using `cmap`.
    linewidth : float
        Contour line width.
    baseline : float
        Lowest contour level = baseline × noise floor. Matches
        pick_peaks so contour base and picking threshold stay consistent.
    figsize : tuple
        Figure size passed to plt.subplots when ax is None.
    ax : matplotlib Axes or None
        If provided, plot into this axes instead of creating a new figure.

    Use `plot_peaklist` to overlay peak markers/labels on the returned axes —
    this lets you plot multiple peaklists on top of the same spectrum.
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


def plot_peaklist(ax, peaks_df, marker="x", peaks_xcol="H_ppm", peaks_ycol="N_ppm",
                  color="limegreen", markersize=3,
                  text="ref_index", label_fontsize=6,
                  label=None,
                  hue=None, palette=None):
    """
    Overlay peak markers (and optional labels) onto an existing spectrum axes.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes to plot onto, e.g. as returned by `plot_spectrum`.
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

    if peaks_xcol not in peaks_df.columns or peaks_ycol not in peaks_df.columns:
        return ax

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

    return ax
