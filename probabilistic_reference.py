import pandas as pd
import numpy as np

ref = pd.read_csv('jardetsky_distns_2009.csv')
REFERENCE_TABLE = {}

for idx, row in ref.iterrows():
    residue = row['AA'].upper()
    ss = row['SS']
    atom = row['Atom_name'].upper()
    mean = row['mean']
    std = row['stdev']

    if residue not in REFERENCE_TABLE:
        REFERENCE_TABLE[residue] = {}
    if ss not in REFERENCE_TABLE[residue]:
        REFERENCE_TABLE[residue][ss] = {}
    REFERENCE_TABLE[residue][ss][atom] = (mean, std)

def gaussian(x, mu, sigma):
    if sigma == 0:
        return 0.0  # or handle as special case
    prefactor = 1 / (sigma * sqrt(2 * pi))
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return prefactor * exp(exponent)


def get_ss_probs(row):
    scores=[]
    for ss in ['C', 'H', 'E']:
        mean, std = REFERENCE_TABLE[row['Comp_ID']][ss][row['Atom_ID']]
        scores.append(gaussian(row['Val'], mean, std))
        
    probs = scores / np.sum(scores)
    selected = ['C','H','E'][np.argmax(scores)]
    return probs, selected

def get_offset(row, corresponding_ha):
    if len(corresponding_ha)==0:
        return NaN
    else:
        ss = corresponding_ha.iloc[0]['ss_max']
        return row['Val'] - REFERENCE_TABLE[row['Comp_ID']][ss][row['Atom_ID']][0]

def apply_offset(row,offsets):
    if row['Atom_ID'] in offsets.keys():
        return row['Val'] - offsets[row['Atom_ID']]
    else:
        return row['Val']
    
def rereference(df):
    '''
    This runs simple version of probablistic re-referencing described by Wishart lab (2005, 2010).
    Original values are copied to `orig` column in dataframe. New values are in 'Val'.

    example code to evaluate referencing on 3 examples

    inds=[4527,6586,4150]
    figure(figsize=(10,2))
    colors = sns.color_palette()
    for j, ind in enumerate(inds):

        # if nmrstar file from bmrb is not downloaded
        # fetch_nmrstar_file(ind) 

        df = get_chem_shifts(parse_nmr_star(f'bmr{ind}_3.str'))
        df = df.loc[df.Atom_ID.isin(['H','HA','N','CA','CB'])]
        df = rereference(df)

        #compare distributions before and after
        for i,atom_id in enumerate(['HA','H','N','CA','CB']):
            subplot(1,5,i+1)
            sns.kdeplot(df.loc[df.Atom_ID==atom_id]['Val'],color=colors[j],label=ind)
            sns.kdeplot(df.loc[df.Atom_ID==atom_id]['orig'],color=colors[j],linestyle=':')
            xlabel(f'omega {atom_id[0]} (ppm)')
            title(atom_id)
            if i==0:
                legend()
    tight_layout()
    '''

    df['orig'] = df['Val'].copy()
    df[['ss_probs','ss_max']] = df.apply(lambda row: get_ss_probs(row), axis=1, result_type='expand')
    ha = df.loc[df.Atom_ID=='HA']
    df['offset'] = df.apply(lambda row: get_offset(row, ha.loc[ha.Seq_ID==row['Seq_ID']]),axis=1)
    
    offsets={}
    
    for i, atom in enumerate(['H','N','CA','CB']):
        vals = np.array(df.loc[df.Atom_ID==atom]['offset'])
        s = np.nanstd(vals)
        m = np.nanmedian(vals)
        vals[np.where(vals>m+3*s)] = nan
        vals[np.where(vals<m-3*s)] = nan
        offsets[atom] = np.nanmedian(vals)
        
    print(offsets)
    df['Val'] = df.apply(lambda row: apply_offset(row, offsets), axis=1)
    return df