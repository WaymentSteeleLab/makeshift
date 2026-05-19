import numpy as np
from .tables import get_panav_distns

REFERENCE_TABLE = get_panav_distns()

def gaussian(x, mu, sigma):
    if sigma == 0:
        return 0.0
    prefactor = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return prefactor * np.exp(exponent)

def get_ss_probs(row):
    scores=[]
    for ss in ['C', 'H', 'E']:
        mean, std = REFERENCE_TABLE[row['Comp_ID']][ss][row['Atom_ID']]
        scores.append(gaussian(row['Val'], mean, std))
    probs = scores / np.sum(scores)
    selected = ['C','H','E'][np.argmax(scores)]
    return probs, selected

def get_offset_panav(row, corresponding_ha):
    if len(corresponding_ha)==0:
        return np.nan
    else:
        ss = corresponding_ha.iloc[0]['ss_max']
        return row['Val'] - REFERENCE_TABLE[row['Comp_ID']][ss][row['Atom_ID']][0]

def apply_offset(row, offsets):
    if row['Atom_ID'] in offsets.keys():
        return row['Val'] - offsets[row['Atom_ID']]
    else:
        return row['Val']
