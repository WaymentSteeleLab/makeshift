"""
PANAV re-referencing offsets for BMRB 5363 (Wang & Wishart 2005 + Wang et al. 2010 CONA).
"""

import numpy as np
import pytest

from makeshift.reref import apply_offsets, compute_offsets

EXPECTED_CHECK = {"N": True, "CA": True, "CB": True, "C": False}


@pytest.fixture(scope="module")
def panav(shifts_5363):
    return compute_offsets(shifts_5363, method="panav")


def test_check_flags(panav):
    _, check, _ = panav
    assert check == EXPECTED_CHECK


def test_offsets_finite(panav):
    offsets, check, _ = panav
    for atom, ok in check.items():
        if ok:
            assert offsets[atom] is not None
            assert np.isfinite(offsets[atom])
        else:
            assert offsets[atom] is None


def test_offsets_are_applied(shifts_5363, panav):
    offsets, _, _ = panav
    corrected = apply_offsets(shifts_5363, offsets)
    assert not corrected["Val"].equals(shifts_5363["Val"])


def test_panav_disagrees_with_lacs(shifts_5363, panav):
    offsets, _, _ = panav
    lacs_offsets, _, _ = compute_offsets(shifts_5363, method="lacs")
    assert offsets["CA"] != pytest.approx(lacs_offsets["CA"], abs=1e-6)


def test_cona_summary(panav):
    _, _, cona = panav
    assert cona is not None
    assert "overall" in cona
    assert set(cona["overall"]) >= {"selected", "confirmed", "score"}
    for k in (3, 4, 5, 6):
        key = f"{k}-residue"
        assert key in cona
        assert cona[key]["selected"] >= cona[key]["confirmed"]
    if cona["overall"]["selected"]:
        assert cona["overall"]["score"] == pytest.approx(
            100.0 * cona["overall"]["confirmed"] / cona["overall"]["selected"]
        )
