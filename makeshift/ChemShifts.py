import pandas as pd
import numpy as np
from .parsing import convert_loop_to_dataframe
from .ReRef import reref_panav_, reref_lacs_
from .utils.chemshift_utils import *

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
    out = out.reset_index()
    out['Seq_ID'] = out['Seq_ID'].astype(int)
    
    if calc_CSI:

        if 'CA' and 'CB' not in out['Atom_ID'].unique():
            raise RuntimeError('Cannot find CA and CB in Atom_ID, cannot calc CSI')
        
        out['csi'] = out.apply(lambda row: get_csi(row, out), axis=1)

    if reref=='panav':
        out = reref_panav_(out)
    elif reref=='lacs':
        out = reref_lacs_(out)

    return out
