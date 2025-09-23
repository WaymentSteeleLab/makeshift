import pandas as pd
import numpy as np
from .utils.reref_utils import *
from .utils.chemshift_utils import *

def reref_panav_(df_0,n_iters=2):
    '''
    This runs simple version of probablistic re-referencing described by Wishart lab (2005, 2010).
    Original values are copied to `orig` column in dataframe. New values are in 'Val'.

    example: evaluate referencing on 3 examples. 4527 should be referenced fine, the others are off

    inds=[4527,6586,4150]
    plt.figure(figsize=(10,2))
    colors = sns.color_palette()
    for j, ind in enumerate(inds):

        # if nmrstar file from bmrb is not downloaded
        fetch_nmrstar_file(ind) 

        df = ms.get_chem_shifts(ms.parse_nmr_star(f'bmr{ind}_3.str'), reref = 'panav')
        print(df.attrs['PANAV offsets'])

        #compare distributions before and after
        for i,atom_id in enumerate(['N','CA','CB']):
            plt.subplot(1,3,i+1)
            sns.kdeplot(df.loc[df.Atom_ID==atom_id]['Val'],color=colors[j],label=ind)
            sns.kdeplot(df.loc[df.Atom_ID==atom_id]['orig'],color=colors[j],linestyle=':')
            plt.xlabel(f'omega {atom_id[0]} (ppm)')
            plt.title(atom_id)
            if i==0:
                plt.legend()
    plt.tight_layout()

    '''
    df = df_0.copy()
    rest = df.loc[~df.Atom_ID.isin(['N','CA','CB'])]

    df = df.loc[df.Atom_ID.isin(['HA','HA2','H','N','CA','CB'])]
    df["Atom_ID"] = df["Atom_ID"].replace("HA2", "HA") # for glycines

    df['orig'] = df['Val'].copy()
    net_offsets={}
    for j in range(n_iters):
        df[['ss_probs','ss_max']] = df.apply(lambda row: get_ss_probs(row), axis=1, result_type='expand')
        ha = df.loc[df.Atom_ID=='HA']
        df['offset'] = df.apply(lambda row: get_offset_panav(row, ha.loc[ha.Seq_ID==row['Seq_ID']]),axis=1)
    
        offsets={}
    
        for i, atom in enumerate(['N','CA','CB']):
            vals = np.array(df.loc[df.Atom_ID==atom]['offset'])
            s = np.nanstd(vals)
            m = np.nanmedian(vals)
            vals[np.where(vals>m+3*s)] = np.nan
            vals[np.where(vals<m-3*s)] = np.nan
            offsets[atom] = np.nanmedian(vals)
            if j==0:
                net_offsets[atom] = np.nanmedian(vals)

            elif j>0:
                net_offsets[atom] += np.nanmedian(vals)
            
            
        df['Val'] = df.apply(lambda row: apply_offset(row, offsets), axis=1)

    df = pd.concat([df, rest])
        
    if df.attrs is None:
        df.attrs = {'PANAV offsets': net_offsets}
    else:
        df.attrs.update({'PANAV offsets': net_offsets})


    return df

############### LACS #################

def reref_lacs_(df_0):
    '''
    LACS-based re-referencing for protein chemical shifts.
    Based on Wang & Markley (2009).
    
    Original values copied to `orig` column. Corrected values in `Val`.
    '''
    df = df_0.copy()
    rest = df.loc[~df.Atom_ID.isin(['N','CA','CB'])]

    df = df.loc[df.Atom_ID.isin(['N','CA','CB'])]

    df['orig'] = df['Val'].copy()

    df['secondary_shift'] = df.apply(lambda row: get_secondary_shift(row), axis=1)
    df['csi'] = df.apply(lambda row: get_csi(row, df), axis=1)
    df['x_ref'] = df.apply(lambda row: get_other_csi(row, df, -1),axis=1)

    offsets={}
    net_offsets = {}
    
    for i,atom in enumerate(['CA', 'CB']):
        df_atom = df[df['Atom_ID'] == atom].copy()

        x_ref_range = chop_at_jumps(df_atom['x_ref'].values)
        ss_range = chop_at_jumps(df_atom['secondary_shift'].values)

        df_atom = df_atom.loc[df_atom.x_ref<=x_ref_range[1]]
        df_atom = df_atom.loc[df_atom.x_ref>=x_ref_range[0]]
        df_atom = df_atom.loc[df_atom.secondary_shift<=ss_range[1]]
        df_atom = df_atom.loc[df_atom.secondary_shift>=ss_range[0]]

        offset = robust_fit(df_atom['x_ref'].values, df_atom['secondary_shift'].values)
        offsets[atom] = offset
        net_offsets[atom] = offset
            
    df['Val'] = df.apply(lambda row: apply_offset(row, offsets), axis=1)

    #repeat again to use re-referenced CA CB for N
    df['secondary_shift'] = df.apply(lambda row: get_secondary_shift(row), axis=1)
    df['csi'] = df.apply(lambda row: get_csi(row, df), axis=1)
    df['x_ref'] = df.apply(lambda row: get_other_csi(row, df, -1),axis=1)

    for i,atom in enumerate(['CA','CB','N']):
        df_atom = df[df['Atom_ID'] == atom].copy() # bound to remove residues that are wrapped 

        x_ref_range = chop_at_jumps(df_atom['x_ref'].values)
        df_atom = df_atom.loc[df_atom.x_ref<=x_ref_range[1]]
        df_atom = df_atom.loc[df_atom.x_ref>=x_ref_range[0]]
        
        offset = robust_fit(df_atom['x_ref'].values, df_atom['secondary_shift'].values)

        if atom in ['CA','CB']:
            offsets[atom] = offset
            net_offsets[atom] += offset
        else:
            offsets[atom] = offset
            net_offsets[atom] = offset            

    df['Val'] = df.apply(lambda row: apply_offset(row, offsets), axis=1)

    df = pd.concat([df, rest])

    net_offsets['method'] = 'LACS'
    if df.attrs is None:
        df.attrs = {'LACS offsets': net_offsets}
    else:
        df.attrs.update({'LACS offsets': net_offsets})


    return df

