"""
PANAV re-referencing offsets for BMRB 5363.

Cumulative round-1 + round-2 offsets, as reported in reref.ipynb.
"""

import pytest

from makeshift.reref import apply_offsets, compute_offsets

EXPECTED_OFFSETS = {
    "N": 0.36842696629213445,
    "CA": -2.1608505747126463,
    "CB": -1.4091951219512187,
    "C": None,
}

EXPECTED_CHECK = {"N": True, "CA": True, "CB": True, "C": False}


@pytest.fixture(scope="module")
def panav(shifts_5363):
    offsets, check = compute_offsets(shifts_5363, method="panav")
    return offsets, check


def test_check_flags(panav):
    _, check = panav
    assert check == EXPECTED_CHECK


@pytest.mark.parametrize("atom, expected", sorted(EXPECTED_OFFSETS.items()))
def test_offsets_match(panav, atom, expected):
    offsets, _ = panav
    assert atom in offsets
    if expected is None:
        assert offsets[atom] is None
    else:
        assert offsets[atom] == pytest.approx(expected, abs=1e-6)


def test_offsets_are_applied(shifts_5363, panav):
    offsets, _ = panav
    corrected = apply_offsets(shifts_5363, offsets)
    assert not corrected["Val"].equals(shifts_5363["Val"])


def test_panav_disagrees_with_lacs(shifts_5363, panav):
    """The two methods fit independently; they shouldn't land on the same CA."""
    offsets, _ = panav
    lacs_offsets, _ = compute_offsets(shifts_5363, method="lacs")
    assert offsets["CA"] != pytest.approx(lacs_offsets["CA"], abs=1e-6)
