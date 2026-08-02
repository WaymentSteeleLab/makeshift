"""
Regression tests for makeshift.hydronmr.engine.run().

Pins the per-residue T1/T2 results against the ground-truth proteins. If
these move, either something broke or the expected values need updating
deliberately.

The ground-truth PDBs aren't in the repo, so these skip unless GT_DIR is
present; point HYDRONMR_GROUND_TRUTH at it to run them.

YJBJ (23001 atoms) is excluded — the dense O(N^2) mobility matrix OOMs on
it and needs the shell/minibead model first.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest

from makeshift.hydronmr.engine import run

GT_DIR = Path(
    os.environ.get(
        "HYDRONMR_GROUND_TRUTH",
        Path(__file__).resolve().parents[2] / "GROUND_TRUTH_DONT_OVERWRITE",
    )
)

# (protein, n_residues, mean T1/T2, std T1/T2)
# Only CYPA (smallest, 1265 atoms) runs here, to keep the suite fast. The
# full 7-protein table is in git history.
EXPECTED = [
    ("CYPA", 164, 7.116249035121523, 0.23293971322837126),
]


@pytest.mark.parametrize("protein, n_expected, mean_expected, std_expected", EXPECTED)
def test_t1_over_t2_regression(protein, n_expected, mean_expected, std_expected, tmp_path):
    pdb_path = GT_DIR / protein / "in.pdb"
    if not pdb_path.exists():
        pytest.skip(f"ground truth PDB not found: {pdb_path}")

    csv_path = tmp_path / f"{protein}_t1t2.csv"
    result = run(pdb_path, csv_path=csv_path)

    ratios = np.array([v[2] for v in result.per_residue.values()])

    assert len(ratios) == n_expected
    assert math.isclose(ratios.mean(), mean_expected, rel_tol=1e-6)
    assert math.isclose(ratios.std(), std_expected, rel_tol=1e-6)

    # CSV was written and matches the in-memory results
    assert csv_path.exists()
    with open(csv_path) as f:
        lines = f.read().strip().splitlines()
    assert lines[0] == "resseq,T1,T2,T1_over_T2,NOE"
    assert len(lines) - 1 == n_expected


@pytest.mark.parametrize("protein", ["CYPA"])
def test_per_residue_values_finite_and_positive(protein):
    pdb_path = GT_DIR / protein / "in.pdb"
    if not pdb_path.exists():
        pytest.skip(f"ground truth PDB not found: {pdb_path}")

    result = run(pdb_path)

    for (chain, resseq), (t1, t2, ratio, noe) in result.per_residue.items():
        assert t1 > 0
        assert t2 > 0
        assert math.isclose(ratio, t1 / t2, rel_tol=1e-9)
        assert all(math.isfinite(v) for v in (t1, t2, ratio, noe))
