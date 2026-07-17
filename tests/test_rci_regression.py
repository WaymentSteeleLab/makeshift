"""
Regression test for makeshift.rci.RCI against the reference implementation
(rci_v_1c.py, Berjanskii/Wishart 2005) on its own bundled test case
(BMRB 4403 / PyJCScorr, the J domain of murine polyomavirus T antigen).

tests/data/rci/PyJCScorr is an old-style (pre-3.0) NMR-STAR file that
makeshift.entry.NMRStarEntry can't parse, so this test reads the sequence
and chemical-shift loop directly rather than going through NMRStarEntry.
"""

from pathlib import Path

import pandas as pd
import pytest

from makeshift.rci import RCI

DATA_DIR = Path(__file__).parent / "data" / "rci"

SEQUENCE = (
    "MDRVLSRADKERLLELLKLPRQLWGDFGRMQQAYKQQSLLLHPDKGGSHALMQELNSLWGTFKTEVYNLRMNLGGTGFQHHHHHH"
)


def _parse_shift_loop(path):
    rows = []
    in_loop = False
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "_Chem_shift_ambiguity_code":
            in_loop = True
            continue
        if in_loop and parts[0] == "stop_":
            break
        if in_loop and len(parts) >= 7:
            try:
                seq_id, comp_id, atom_id, val = int(parts[1]), parts[2], parts[3], float(parts[5])
            except ValueError:
                continue
            rows.append((seq_id, comp_id, atom_id, val))
    return pd.DataFrame(rows, columns=["Seq_ID", "Comp_ID", "Atom_ID", "Val"])


def test_rci_matches_reference_output():
    shifts = _parse_shift_loop(DATA_DIR / "PyJCScorr")
    expected = pd.read_csv(
        DATA_DIR / "PyJCScorr_RCI.txt_compare", sep=" ", header=None,
        names=["Seq_ID", "RCI", "Comp_ID"],
    )

    r = RCI(shifts, sequence=SEQUENCE, first_resid=1)
    r.run()

    merged = r.results[["Seq_ID", "RCI"]].merge(expected[["Seq_ID", "RCI"]],
                                                  on="Seq_ID", suffixes=("_mine", "_ref"))
    assert len(merged) == len(expected)
    assert (merged["RCI_mine"] - merged["RCI_ref"]).abs().max() == pytest.approx(0, abs=1e-6)


def test_rci_talosn_algorithm_is_pinned():
    """
    algorithm="talosn" (makeshift.rci._talosn) was validated against the
    real compiled TALOS-N v4.11 binary two ways: a debug-instrumented
    rebuild of RCI.cpp confirmed exact intermediate values on a spot
    check, and a 9-BMRB-entry comparison against TalosN's real predS2.tab
    output gave pooled r=0.989, median abs S2 diff=0.0019 (remaining
    error is dominated by TALOS.cpp's calcAverageCS, an unported
    homology-database nearest-neighbor predictor for isolated missing
    atoms -- see the "Known remaining gap" note in makeshift/rci/_talosn.py
    -- not a bug in what is ported).

    There's no real TALOS-N output committed for this particular fixture
    (BMRB 4403 / PyJCScorr) to compare against -- the 9 entries used for
    that validation are gitignored .str downloads -- so this test instead
    pins the current, already-validated output against a snapshot to
    catch regressions in the port itself.
    """
    shifts = _parse_shift_loop(DATA_DIR / "PyJCScorr")
    expected = pd.read_csv(DATA_DIR / "PyJCScorr_talosn_expected.csv")

    r = RCI(shifts, sequence=SEQUENCE, first_resid=1, algorithm="talosn")
    r.run()

    merged = r.results[["Seq_ID", "RCI", "S2"]].merge(
        expected, on="Seq_ID", suffixes=("_mine", "_ref")
    )
    assert len(merged) == len(expected)
    assert (merged["RCI_mine"] - merged["RCI_ref"]).abs().max() == pytest.approx(0, abs=1e-9)
    assert (merged["S2_mine"] - merged["S2_ref"]).abs().max() == pytest.approx(0, abs=1e-9)
