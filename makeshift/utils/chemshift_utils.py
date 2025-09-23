import numpy as np
from .tables import get_random_coil

RANDOM_COIL = get_random_coil()

def get_secondary_shift(row):
    """Calculate secondary shift (observed - random coil)"""
    return row['Val'] - RANDOM_COIL[row['Comp_ID']][row['Atom_ID']]

def get_csi(row, df):
    data = df.loc[df.Seq_ID==row['Seq_ID']]
    ca = data[data['Atom_ID'] == 'CA']
    cb = data[data['Atom_ID'] == 'CB']
    if len(ca) >= 1 and len(cb) >= 1:
        ca_sec = get_secondary_shift(ca.iloc[0])
        cb_sec = get_secondary_shift(cb.iloc[0])

        if not (np.isnan(ca_sec) or np.isnan(cb_sec)):
            return ca_sec - cb_sec
        else:
            return np.nan
    else:
        return np.nan

def get_other_csi(row, df, index):
    """Get (CA-CB) secondary shift difference from other residue"""
    data = df[df['Seq_ID'] == row['Seq_ID'] + index]
    if len(data)>=1:
        return data['csi'].iloc[0]
    else:
        return np.nan