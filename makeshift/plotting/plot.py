import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from ..utils.plot_utils import pivot_chem_shifts


def plot_spectrum(cs_wide, type='hsqc', ax=None, label=True, **scatter_kwargs):
    """Plot an NMR spectrum from wide-format chem shift data.

    Parameters
    ----------
    cs_wide : DataFrame
        Wide-format DataFrame from pivot_chem_shifts(), with columns H, N, Seq_ID, Comp_ID.
    type : str
        Spectrum type. Currently supports 'hsqc'.
    ax : matplotlib Axes, optional
        Axes to plot on. Creates a new figure if None.
    label : bool
        Annotate each peak with residue name + number.
    **scatter_kwargs
        Passed to ax.scatter (e.g. color, s, alpha).

    Returns
    -------
    fig, ax
    """
    if type != 'hsqc':
        raise ValueError(f"Unsupported spectrum type '{type}'. Supported: 'hsqc'")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    scatter_kwargs.setdefault('color', 'steelblue')
    scatter_kwargs.setdefault('s', 40)
    scatter_kwargs.setdefault('zorder', 3)

    ax.scatter(cs_wide['H'], cs_wide['N'], **scatter_kwargs)

    if label:
        for _, row in cs_wide.iterrows():
            ax.text(
                row['H'], row['N'],
                f"{row['Comp_ID']}{row['Seq_ID']}",
                fontsize=7,
                alpha=0.85,
            )

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel('$^1$H (ppm)')
    ax.set_ylabel('$^{15}$N (ppm)')

    return fig, ax


def plot_hsqc_comparison(cs_wide, pred_df, ax=None, label=True):
    """Overlay experimental HSQC peaks with UCBShift predictions.

    Experimental peaks are plotted as circles, predicted as X markers.
    Connecting lines are drawn between matched residues, colored per residue.

    Parameters
    ----------
    cs_wide : DataFrame
        Wide-format experimental data from pivot_chem_shifts(), with columns
        H, N, Seq_ID, Comp_ID.
    pred_df : DataFrame
        Predictions from UCBShift with columns RESNUM, H_X, N_X.
    ax : matplotlib Axes, optional
        Axes to plot on. Creates a new figure if None.
    label : bool
        Annotate experimental peaks with residue labels.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    all_ids = sorted(set(cs_wide['Seq_ID']).union(set(pred_df['RESNUM'])))
    colors = {k: cm.rainbow(i / max(len(all_ids) - 1, 1)) for i, k in enumerate(all_ids)}

    # experimental peaks
    for _, row in cs_wide.iterrows():
        c = colors[row['Seq_ID']]
        ax.scatter(row['H'], row['N'], color=c, s=40, zorder=3)
        if label:
            ax.text(
                row['H'], row['N'],
                f"{row['Comp_ID']}{row['Seq_ID']}",
                fontsize=7,
                alpha=0.85,
            )

    # predicted peaks
    for _, row in pred_df.iterrows():
        c = colors.get(row['RESNUM'], 'gray')
        ax.scatter(row['H_X'], row['N_X'], color=c, marker='x', s=40, zorder=3)

    # connecting lines between matched residues
    shared = set(cs_wide['Seq_ID']).intersection(set(pred_df['RESNUM']))
    for k in shared:
        exp_rows = cs_wide[cs_wide['Seq_ID'] == k]
        pred_rows = pred_df[pred_df['RESNUM'] == k]
        c = colors[k]
        for _, r1 in exp_rows.iterrows():
            for _, r2 in pred_rows.iterrows():
                ax.plot(
                    [r1['H'], r2['H_X']],
                    [r1['N'], r2['N_X']],
                    color=c,
                    alpha=0.5,
                    linewidth=1,
                )

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel('$^1$H (ppm)')
    ax.set_ylabel('$^{15}$N (ppm)')

    return fig, ax
