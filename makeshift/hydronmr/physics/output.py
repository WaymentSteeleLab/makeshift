"""
Result output: per-residue T1/T2/NOE as CSV.

Columns: resseq, T1, T2, T1_over_T2, NOE
"""

import csv
from pathlib import Path
from typing import Dict, Tuple

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..engine import HydronmrState
from .nmr import per_residue_t1t2


def per_residue_results(g: "HydronmrState", nh_vectors: Dict[Tuple[str, int], "object"]):
    """Compute (resseq -> (T1, T2, T1/T2, NOE)) for every residue with a
    valid N-H bond vector. `nh_vectors` is the dict returned by
    `routines.pdb.nh_bond_vectors`, keyed by (chain, resseq)."""
    results = {}
    for (chain, resseq), v in nh_vectors.items():
        t1, t2, ratio, noe = per_residue_t1t2(g, v)
        results[(chain, resseq)] = (t1, t2, ratio, noe)
    return results


def write_t1t2_csv(g: "HydronmrState", nh_vectors, path) -> Path:
    """Write per-residue T1/T2/NOE results to a CSV file at `path`.

    Returns the path written.
    """
    results = per_residue_results(g, nh_vectors)
    path = Path(path)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["resseq", "T1", "T2", "T1_over_T2", "NOE"])
        for (chain, resseq) in sorted(results, key=lambda k: (k[0], k[1])):
            t1, t2, ratio, noe = results[(chain, resseq)]
            writer.writerow([resseq, f"{t1:.6g}", f"{t2:.6g}", f"{ratio:.6g}", f"{noe:.6g}"])
    return path
