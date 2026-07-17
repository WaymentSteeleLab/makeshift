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


_RCI_DATA = _DATA / 'rci_data'

# Valid choices for get_rci_tables(neighbor_table=...). Each is a distinct
# set of preceed/next-residue correction values the reference script
# (rci_v_1c.py) could be configured to use via its neighbor_flag /
# sw_neighbor_flag switches:
#   "schwarzinger"  -- the script's own default (neighbor_flag=1)
#   "wang"          -- neighbor_flag=0
#   "schwartz_wang" -- sw_neighbor_flag=1 (a Schwarzinger/Wang hybrid)
RCI_NEIGHBOR_TABLES = ("schwarzinger", "wang", "schwartz_wang")


def _load_rci_wide_table(filename):
    """RCI atom-effect table: residue -> {N,CO,CA,CB,NH,HA: float}, NaN where undefined."""
    df = pd.read_csv(_RCI_DATA / filename, index_col='residue')
    return df.astype(float)


def _load_rci_neighbor_table(table_name, neighbor_table, ss="C"):
    """
    A preceed/next-residue correction table, pivoted from its long-format
    (residue, atom, ss, value) CSV down to a single secondary-structure
    slice (default "C" = coil, matching every RCI run today -- none of
    the callers currently predict secondary structure, so the "H"/"B"
    slices exist in the CSVs for completeness but aren't reachable yet).
    """
    df = pd.read_csv(_RCI_DATA / f"{table_name}_{neighbor_table}.csv")
    df = df[df["ss"] == ss]
    return df.pivot(index="residue", columns="atom", values="value")


def get_talosn_rc_tables():
    """
    TALOS-N's own random-coil-adjustment tables (randcoil.tab, rcadj.tab,
    rcprev.tab, rcnext.tab, bundled with the TALOS-N binary) -- a
    different table system from RCI's own Schwarzinger tables above, used
    by TALOS-N's RCI-S2 module (RCI.cpp) to synthesize a value for atoms
    that were never observed (see makeshift.rci._talosn). Each is a
    DataFrame indexed by one-letter residue code (plus "c" for TALOS-N's
    own oxidized-Cys variant, its own >=34.0ppm CB threshold -- distinct
    from RCI's 35.0ppm one) with columns [N, CO, CA, CB, NH, HA]; NaN
    marks an atom with no defined value (Gly CB, Pro NH in randcoil only).

    Returns a dict with keys: randcoil, rcadj, rcprev, rcnext.
    """
    return {
        "randcoil": _load_rci_wide_table("talosn_randcoil.csv"),
        "rcadj": _load_rci_wide_table("talosn_rcadj.csv"),
        "rcprev": _load_rci_wide_table("talosn_rcprev.csv"),
        "rcnext": _load_rci_wide_table("talosn_rcnext.csv"),
    }


def get_rci_tables(neighbor_table="schwarzinger"):
    """
    Lookup tables for the RCI (Random Coil Index) flexibility predictor.
    `neighbor_table` selects which preceed/next-residue correction values
    to use -- see RCI_NEIGHBOR_TABLES for the options and what they mean.
    Each returned value is a DataFrame indexed by one-letter residue code
    with columns [N, CO, CA, CB, NH, HA]; NaN marks an atom that has no
    defined value for that residue (e.g. Gly CB, Pro N/NH).

    Returns a dict with keys: random_coil, preceed_effect, next_effect,
    preceed_preceed_effect, next_next_effect.
    """
    if neighbor_table not in RCI_NEIGHBOR_TABLES:
        raise ValueError(
            f"neighbor_table={neighbor_table!r} not recognized; "
            f"choose one of {RCI_NEIGHBOR_TABLES}"
        )
    return {
        "random_coil": _load_rci_wide_table("random_coil.csv"),
        "preceed_effect": _load_rci_neighbor_table("preceed_effect", neighbor_table),
        "next_effect": _load_rci_neighbor_table("next_effect", neighbor_table),
        "preceed_preceed_effect": _load_rci_wide_table("preceed_preceed_effect.csv"),
        "next_next_effect": _load_rci_wide_table("next_next_effect.csv"),
    }
