"""
Regression tests for hydronmr.engine.run().

Pins the per-residue T1/T2 results (mean, std, and Pearson correlation
against the ground-truth reference) for the 7 small/medium ground-truth
proteins. If these change, either a real bug was introduced or the
expected values need to be updated deliberately (e.g. after task #4 /
shell-model work changes the IND=3 results).

YJBJ (23001 atoms) is excluded: the current dense O(N^2) mobility-matrix
approach OOMs on it (needs the shell/minibead model, task #4).
"""

import math
from pathlib import Path

import numpy as np
import pytest

from hydronmr.engine import run

ROOT = Path(__file__).resolve().parents[2]
GT_DIR = ROOT / "GROUND_TRUTH_DONT_OVERWRITE"

# (protein, n_residues, mean T1/T2, std T1/T2)
# Only CYPA (smallest, 1265 atoms) is exercised here to keep the suite
# fast; see git history for the full 7-protein table if needed.
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
