""" Shared helpers for the reref subpackage. """

from ..utils.constants import _BACKBONE


def apply_offsets(df, offsets):
    """
    Return a copy of ``df`` with ``Val + offset`` per atom
    (``offset ≈ d_ave − d_obs``). Atoms absent from ``offsets`` (or None)
    are left unchanged.
    """
    applied = {a: v for a, v in (offsets or {}).items() if v is not None}
    out = df.copy()
    if applied:
        out["Val"] = out["Val"] + out["Atom_ID"].map(applied).fillna(0.0)
    return out
