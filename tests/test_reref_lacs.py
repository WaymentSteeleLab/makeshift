"""
LACS re-referencing offsets for BMRB 5363.

Expected values are the ones reported in reref.ipynb. GLY HA averaging is
handled inside compute_offsets, so nothing needs doing to the input here.
"""

import pytest

from makeshift.reref import apply_offsets, compute_offsets

EXPECTED_OFFSETS = {
    "N": 1.0884181598620282,
    "CA": -2.121216467191502,
    "CB": -2.1212146555686777,
    "C": None,
    "H": 0.17819534303635887,
}

EXPECTED_CHECK = {"CA": True, "CB": True, "C": False, "N": True, "H": True}


@pytest.fixture(scope="module")
def lacs(shifts_5363):
    offsets, check = compute_offsets(shifts_5363, method="lacs")
    return offsets, check


def test_check_flags(lacs):
    _, check = lacs
    assert check == EXPECTED_CHECK


@pytest.mark.parametrize("atom, expected", sorted(EXPECTED_OFFSETS.items()))
def test_offsets_match(lacs, atom, expected):
    offsets, _ = lacs
    assert atom in offsets
    if expected is None:
        assert offsets[atom] is None
    else:
        assert offsets[atom] == pytest.approx(expected, abs=1e-6)


def test_offsets_are_applied(shifts_5363, lacs):
    offsets, _ = lacs
    corrected = apply_offsets(shifts_5363, offsets)

    assert not corrected["Val"].equals(shifts_5363["Val"])

    ca = shifts_5363["Atom_ID"] == "CA"
    shift = (shifts_5363.loc[ca, "Val"] - corrected.loc[ca, "Val"]).dropna()
    assert shift.round(6).nunique() == 1
    assert shift.iloc[0] == pytest.approx(EXPECTED_OFFSETS["CA"], abs=1e-6)


def test_atoms_without_an_offset_pass_through(shifts_5363, lacs):
    offsets, _ = lacs
    corrected = apply_offsets(shifts_5363, offsets)

    sidechain = ~shifts_5363["Atom_ID"].isin(offsets)
    assert sidechain.any()
    assert corrected.loc[sidechain, "Val"].equals(shifts_5363.loc[sidechain, "Val"])
