import numpy as np
from .tables import get_random_coil

RANDOM_COIL = get_random_coil()

def rc_lookup(comp_id, atom_id):
    ''' Safe random coil evaluation '''
    try:
        return RANDOM_COIL[comp_id.upper()][atom_id.upper()]
    except KeyError:
        return np.nan

def get_secondary_shift(row):
    """Calculate secondary shift (observed - random coil)"""
    rc = rc_lookup(row['Comp_ID'], row['Atom_ID'])
    val = row['Val']
    return np.nan if (rc is None or np.isnan(rc) or np.isnan(val)) else (val - rc)

def get_csi(row, df):

    ''' Calcualate (CA-CB) secondary shift '''

    data = df.loc[df.Seq_ID==row['Seq_ID']]
    ca = data[data['Atom_ID'] == 'CA']
    cb = data[data['Atom_ID'] == 'CB']

    ca_sec = get_secondary_shift(ca.iloc[0]) if len(ca) else np.nan
    cb_sec = get_secondary_shift(cb.iloc[0]) if len(cb) else np.nan

    if np.isfinite(ca_sec) and np.isfinite(cb_sec):
        return ca_sec - cb_sec
    if np.isfinite(ca_sec):
        return ca_sec
    return np.nan


def csi_index(row, helix=0.7, strand=-0.7, gly=0.7):
    """ Map raw value to {-1,0,+1}. """

    value, residue = row["csi_raw"], row['Seq_ID']
    
    if np.isnan(value): return np.nan

    # has CA and CB
    if value >= helix: return +1
    if value <= strand: return -1

    # GLY does not have a CB
    if residue == 'GLY':
        if value >= +gly: return +1
        if value <= -gly: return -1
    return 0


def get_other_csi(row, df, index):
    """Get (CA-CB) secondary shift difference from other residue"""
    data = df[df['Seq_ID'] == row['Seq_ID'] + index]
    if len(data)>=1:
        return data['csi'].iloc[0]
    else:
        return np.nan
