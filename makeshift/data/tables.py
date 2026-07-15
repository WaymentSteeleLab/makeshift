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


def _load_rci_table(filename):
    """RCI atom-effect table: residue -> {N,CO,CA,CB,NH,HA: float}, NaN where undefined."""
    df = pd.read_csv(_DATA / filename, index_col='residue')
    return df.astype(float)


def get_rci_tables():
    """
    Lookup tables for the RCI (Random Coil Index) flexibility predictor
    (Berjanskii/Schwarzinger neighbor-correction scheme). Each value is a
    DataFrame indexed by one-letter residue code with columns
    [N, CO, CA, CB, NH, HA]; NaN marks an atom that has no defined value
    for that residue (e.g. Gly CB, Pro N/NH).

    Returns a dict with keys: random_coil, preceed_effect, next_effect,
    preceed_preceed_effect, next_next_effect.
    """
    return {
        "random_coil": _load_rci_table("rci_random_coil.csv"),
        "preceed_effect": _load_rci_table("rci_preceed_effect.csv"),
        "next_effect": _load_rci_table("rci_next_effect.csv"),
        "preceed_preceed_effect": _load_rci_table("rci_preceed_preceed_effect.csv"),
        "next_next_effect": _load_rci_table("rci_next_next_effect.csv"),
    }
