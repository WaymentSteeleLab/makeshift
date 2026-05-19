import numpy as np
from .tables import get_random_coil

RANDOM_COIL = get_random_coil()

def rc_lookup(comp_id, atom_id):
    try:
        return RANDOM_COIL[comp_id.upper()][atom_id.upper()]
    except KeyError:
        return np.nan

def get_secondary_shift(row):
    """Calculate secondary shift (observed - random coil)."""
    rc = rc_lookup(row['Comp_ID'], row['Atom_ID'])
    val = row['Val']
    return np.nan if (rc is None or np.isnan(rc) or np.isnan(val)) else (val - rc)

def get_csi(row, df, offset=0, strict=False):
    """
    Chemical shift index: (CA - CB) secondary shift.

    Parameters
    ----------
    offset : int
        Residue offset relative to row['Seq_ID'].
        0  — compute from CA/CB of the current residue.
        ±n — look up the pre-computed 'csi' column for the adjacent residue
             (requires 'csi' to already exist in df).
    strict : bool
        Only applies when offset=0.
        False (default) — fall back to CA-only secondary shift when CB is
                          missing, e.g. for GLY. Suitable for visualisation.
        True            — return NaN when CB is missing. Use for LACS fitting
                          to avoid a CA-only value corrupting the regression.
    """
    if offset != 0:
        data = df[df['Seq_ID'] == row['Seq_ID'] + offset]
        return data['csi'].iloc[0] if len(data) >= 1 else np.nan

    data = df.loc[df.Seq_ID == row['Seq_ID']]
    ca = data[data['Atom_ID'] == 'CA']
    cb = data[data['Atom_ID'] == 'CB']

    ca_sec = get_secondary_shift(ca.iloc[0]) if len(ca) else np.nan
    cb_sec = get_secondary_shift(cb.iloc[0]) if len(cb) else np.nan

    if np.isfinite(ca_sec) and np.isfinite(cb_sec):
        return ca_sec - cb_sec
    if not strict and np.isfinite(ca_sec):
        return ca_sec
    return np.nan


def csi_index(row, helix=0.7, strand=-0.7, gly=0.7):
    """Map csi_raw to {-1, 0, +1}."""
    value, residue = row["csi_raw"], row['Seq_ID']
    if np.isnan(value):
        return np.nan
    if value >= helix:
        return +1
    if value <= strand:
        return -1
    if residue == 'GLY':
        if value >= +gly:
            return +1
        if value <= -gly:
            return -1
    return 0
