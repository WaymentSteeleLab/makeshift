"""
HYDRONMR: per-residue NMR relaxation (T1/T2) predicted from a PDB structure.

A self-contained Python port of the HYDRONMR mode-25 calculation. The only
public entry point is :func:`run`; everything else lives under `physics`.

    from makeshift import hydronmr
    result = hydronmr.run("in.pdb", csv_path="t1t2.csv")
"""

from .engine import run

__all__ = ["run"]
