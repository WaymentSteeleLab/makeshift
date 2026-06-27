"""
Re-referencing of protein backbone chemical shifts (LACS and PANAV).
"""

from .base import apply_offsets, _BACKBONE
from .lacs import reref_lacs, _N_STD_OUTLIER
from .panav import reref_panav

__all__ = ["compute_offsets", "apply_offsets"]


def compute_offsets(df, method, n_std=_N_STD_OUTLIER):
    """
    Fit per-atom re-referencing offsets for a long-format shift table.

    Parameters
    ----------
    df : DataFrame
        Long-format shifts with at least Comp_ID, Seq_ID, Atom_ID, Val.
    method : {'lacs', 'panav'}
    n_std : int
        LACS only — BMRB-statistics outlier threshold (default 4).

    Returns
    -------
    offsets : dict {atom: float | None}
        Correction per atom such that ``corrected = Val - offset``; None where
        fitting failed. LACS atoms: CA, CB, C, N, H. PANAV atoms: N, CA, CB, C.
    check : dict {atom: bool}
        Whether re-referencing succeeded for each atom.

    Returns ``(None, None)`` if no backbone shifts are present.
    """
    if method not in ("lacs", "panav"):
        raise ValueError(f"method must be 'lacs' or 'panav', got {method!r}")

    work = df.copy()
    work = work.loc[
        work.Atom_ID.isin(_BACKBONE) | work.Atom_ID.str.contains("^HA", na=False)
    ]

    # average GLY HA/HA2 into one HA per residue
    gly = (work["Comp_ID"] == "GLY") & work["Atom_ID"].str.contains("HA", na=False)
    for seq_id, group in work[gly].groupby("Seq_ID"):
        mask = (work["Seq_ID"] == seq_id) & gly
        work.loc[mask, "Val"] = group["Val"].mean()
        idx = work.loc[mask].index
        work.loc[idx[0], "Atom_ID"] = "HA"
        if len(idx) > 1:
            work.loc[idx[1:], "Atom_ID"] = "HA2"

    work = work.loc[work.Atom_ID.isin(_BACKBONE)].copy()
    if work.empty:
        return None, None

    if method == "lacs":
        return reref_lacs(work, n_std=n_std)

    return reref_panav(work)