import warnings
import numpy as np
import pandas as pd

from .utils.chemshift_utils import *
from .ReRef import reref as _reref
from .parsing import convert_loop_to_dataframe

def clean_cs_dataframe(df):
    df['Val'] = df['Val'].replace('.',np.nan)
    df['Val'] = df['Val'].astype(float)
    
    return df[['Entity_ID', 'Seq_ID', 'Auth_seq_ID','Comp_ID',
                'Atom_ID','Atom_type','Val']]

def get_chem_shifts(parsed, calc_CSI=False, reref=None):

    out_dfs=[]
    for k, entry in parsed['assigned_chemical_shifts'].items():
        cs = convert_loop_to_dataframe(entry['_Atom_chem_shift'])   
        cs = clean_cs_dataframe(cs)
        try:
            cs['name'] = parsed[k]['Name']
        except:
            cs['name'] = '.'
        cs['cs_saveframe_id'] = k
        out_dfs.append(cs)
    
    out = pd.concat(out_dfs)

    out['Seq_ID'] = out['Seq_ID'].astype(int)
    
    if reref in ('panav', 'lacs'):
        out, _, _ = _reref(out, method=reref)

    if calc_CSI:
        if 'CA' and 'CB' not in out['Atom_ID'].unique():
            warnings.warn("Cannot find CA and CB in Atom_ID. Cannot calculate CSI", UserWarning)
        elif not sum(~pd.isna(out['Atom_ID'] == 'CA')):
            warnings.warn("There are only np.nan values for CA. Cannot calculate CSI", UserWarning)
        if not sum(~pd.isna(out['Atom_ID'] == 'CB')):
            warnings.warn("There are only np.nan values for CB. Cannot calculate CSI", UserWarning)
        else:
            out['csi_raw'] = out.apply(lambda row: get_csi(row, out), axis=1)
            out["csi"] = out.apply(lambda row: csi_index(row), axis=1).astype(float)
    
    return out
