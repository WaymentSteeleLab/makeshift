import pytest

BMRB_ID = 5363
ASSIGN_NAME = "chemical_shifts_1"


@pytest.fixture(scope="session")
def shifts_5363():
    """
    The chemical_shifts_1 saveframe of BMRB 5363, as a long-format table.

    Needs network. Skips rather than fails if BMRB is unreachable, so the
    offline part of the suite still runs.
    """
    from makeshift import ChemicalShifts

    try:
        cs = ChemicalShifts.from_bmrb(BMRB_ID)
    except Exception as exc:                      # noqa: BLE001 - network, parse, 404...
        pytest.skip(f"could not fetch BMRB {BMRB_ID}: {exc}")

    df = cs.data[cs.data["ChemShift_ID"] == ASSIGN_NAME].copy()
    if df.empty:
        pytest.skip(f"BMRB {BMRB_ID} has no saveframe {ASSIGN_NAME!r}")
    return df.sort_values("Seq_ID").reset_index(drop=True)
