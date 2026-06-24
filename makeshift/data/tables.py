import numpy as np
import pandas as pd
from pathlib import Path

_DATA = Path(__file__).parent.parent / 'data'


def get_random_coil():
    """Random coil chemical shifts (Wishart & Sykes 1994). Returns {residue: {atom: float}}."""
    df = pd.read_csv(_DATA / 'random_coil.csv')
    out = {}
    for _, row in df.iterrows():
        res, atom = row['residue'], row['atom']
        val = row['value']
        out.setdefault(res, {})[atom] = np.nan if (isinstance(val, float) and np.isnan(val)) or val == '' else float(val)
    return out


def get_panav_distns():
    """PANAV reference distributions. Returns {residue: {ss: {atom: (mean, std)}}}."""
    df = pd.read_csv(_DATA / 'panav_distns.csv')
    out = {}
    for _, row in df.iterrows():
        res  = row['AA'].upper()
        ss   = row['SS']
        atom = row['Atom_name'].upper()
        out.setdefault(res, {}).setdefault(ss, {})[atom] = (float(row['mean']), float(row['stdev']))
    return out


def get_bmrb_stats():
    """BMRB full-database statistics. Returns {residue: {atom: (mean, std)}}."""
    df = pd.read_csv(_DATA / 'bmrb_stats.csv')
    out = {}
    for _, row in df.iterrows():
        out.setdefault(row['residue'], {})[row['atom']] = (float(row['mean']), float(row['std']))
    return out


def get_c_prime_rc():
    """C' random coil values (Wishart et al. 1995). Returns {residue: float}."""
    df = pd.read_csv(_DATA / 'c_prime_rc.csv')
    return dict(zip(df['residue'], df['value'].astype(float)))
