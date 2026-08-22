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


def get_csi_wishart():
    """Wishart CSI ranges. Returns {residue: {atom: (center, half_width)}}."""
    df = pd.read_csv(_DATA / "csi_wishart.csv", comment="#")
    out = {}
    for _, row in df.iterrows():
        out.setdefault(str(row["residue"]), {})[str(row["atom"])] = (
            float(row["center"]), float(row["half_width"]),
        )
    return out


def get_rc_pre_pro():
    """Wishart 1995 Table 5 (X before Pro). Returns {residue: {atom: float}}."""
    df = pd.read_csv(_DATA / "rc_pre_pro.csv", comment="#")
    out = {}
    for _, row in df.iterrows():
        out.setdefault(str(row["residue"]), {})[str(row["atom"])] = float(row["value"])
    return out


def get_rc_n_prev():
    """Wishart 1995 Table 8 (15N of Ala after X). Returns {prev_residue: float}."""
    df = pd.read_csv(_DATA / "rc_n_prev.csv", comment="#")
    return dict(zip(df["prev_residue"].astype(str), df["N_ala"].astype(float)))


RCI_NEIGHBOR_TABLES = ("schwarzinger", "wang", "schwartz_wang")
_RCI_ATOMS = ["N", "CO", "CA", "CB", "NH", "HA"]


def _rci_wide(path):
    df = pd.read_csv(path, comment="#")
    df = df.set_index(df.columns[0])
    return df.apply(pd.to_numeric, errors="coerce")


def _rci_neighbor(path, ss="C"):
    df = pd.read_csv(path, comment="#")
    if "ss" in df.columns:
        df = df[df["ss"] == ss]
    wide = df.pivot_table(index="residue", columns="atom", values="value", aggfunc="first")
    for atom in _RCI_ATOMS:
        if atom not in wide.columns:
            wide[atom] = np.nan
    return wide[_RCI_ATOMS].apply(pd.to_numeric, errors="coerce")


def get_rci_tables(neighbor_table="schwarzinger"):
    """Schwarzinger/Wang neighbor tables for Wishart RCI. Values are coil-state."""
    if neighbor_table not in RCI_NEIGHBOR_TABLES:
        raise ValueError(f"neighbor_table must be one of {RCI_NEIGHBOR_TABLES}, got {neighbor_table!r}")
    root = _DATA / "rci_data"
    return {
        "random_coil": _rci_wide(root / "random_coil.csv"),
        "preceed_effect": _rci_neighbor(root / f"preceed_effect_{neighbor_table}.csv"),
        "next_effect": _rci_neighbor(root / f"next_effect_{neighbor_table}.csv"),
        "preceed_preceed_effect": _rci_wide(root / "preceed_preceed_effect.csv"),
        "next_next_effect": _rci_wide(root / "next_next_effect.csv"),
    }


def get_talosn_rc_tables():
    """TALOS-N randcoil / rcadj / rcprev / rcnext tables."""
    root = _DATA / "rci_data"
    return {
        "randcoil": _rci_wide(root / "talosn_randcoil.csv"),
        "rcadj": _rci_wide(root / "talosn_rcadj.csv"),
        "rcprev": _rci_wide(root / "talosn_rcprev.csv"),
        "rcnext": _rci_wide(root / "talosn_rcnext.csv"),
    }
