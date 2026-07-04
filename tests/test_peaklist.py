import unittest
import warnings

import pandas as pd

from makeshift.peaklist import PeakList, _DEFAULT_DIMS


class _FakeEntry:
    entry_id = "TEST1"


class _FakeChemicalShifts:
    """Minimal stand-in for makeshift.chemshift.ChemicalShifts."""

    def __init__(self, data):
        self.data = data
        self.entry = _FakeEntry()


def _cs(rows):
    return _FakeChemicalShifts(pd.DataFrame(rows))


# Two residues with N, H/HN, CA, CB shifts; residue 2 has no CB (to test
# partial-dimension coverage).
_ROWS = [
    {"ChemShift_ID": "sf1", "Entity_ID": 1, "Seq_ID": 1, "Auth_seq_ID": 1,
     "Comp_ID": "ALA", "Atom_ID": "N", "Val": 120.0},
    {"ChemShift_ID": "sf1", "Entity_ID": 1, "Seq_ID": 1, "Auth_seq_ID": 1,
     "Comp_ID": "ALA", "Atom_ID": "H", "Val": 8.1},
    {"ChemShift_ID": "sf1", "Entity_ID": 1, "Seq_ID": 1, "Auth_seq_ID": 1,
     "Comp_ID": "ALA", "Atom_ID": "CA", "Val": 52.0},
    {"ChemShift_ID": "sf1", "Entity_ID": 1, "Seq_ID": 1, "Auth_seq_ID": 1,
     "Comp_ID": "ALA", "Atom_ID": "CB", "Val": 19.0},
    {"ChemShift_ID": "sf1", "Entity_ID": 1, "Seq_ID": 2, "Auth_seq_ID": 2,
     "Comp_ID": "GLY", "Atom_ID": "N", "Val": 110.0},
    {"ChemShift_ID": "sf1", "Entity_ID": 1, "Seq_ID": 2, "Auth_seq_ID": 2,
     "Comp_ID": "GLY", "Atom_ID": "HN", "Val": 8.5},
    {"ChemShift_ID": "sf1", "Entity_ID": 1, "Seq_ID": 2, "Auth_seq_ID": 2,
     "Comp_ID": "GLY", "Atom_ID": "CA", "Val": 45.0},
]


class TestFromChemshiftsDefaultDims(unittest.TestCase):
    """Backbone-amide (H, N) is still the default and behaves as before."""

    @classmethod
    def setUpClass(cls):
        cls.pl = PeakList.from_chemshifts(_cs(_ROWS))

    def test_columns(self):
        self.assertEqual(
            list(self.pl.data.columns),
            ["Seq_ID", "Auth_seq_ID", "Comp_ID", "H_ppm", "N_ppm", "assn_label"],
        )

    def test_values(self):
        row = self.pl.data.set_index("Seq_ID").loc[1]
        self.assertEqual(row["H_ppm"], 8.1)
        self.assertEqual(row["N_ppm"], 120.0)
        self.assertEqual(row["assn_label"], "A1")

    def test_hn_alias_accepted(self):
        # residue 2 uses "HN" rather than "H" for the amide proton
        row = self.pl.data.set_index("Seq_ID").loc[2]
        self.assertEqual(row["H_ppm"], 8.5)
        self.assertEqual(row["assn_label"], "G2")

    def test_two_residues(self):
        self.assertEqual(len(self.pl.data), 2)


class TestFromChemshiftsCustomDims(unittest.TestCase):

    def test_3d_hnca(self):
        dims = (("H", ("H", "HN")), ("N", ("N",)), ("CA", ("CA",)))
        pl = PeakList.from_chemshifts(_cs(_ROWS), dims=dims)
        self.assertEqual(
            list(pl.data.columns),
            ["Seq_ID", "Auth_seq_ID", "Comp_ID", "H_ppm", "N_ppm", "CA_ppm", "assn_label"],
        )
        self.assertEqual(len(pl.data), 2)
        self.assertEqual(pl.data.set_index("Seq_ID").loc[1, "CA_ppm"], 52.0)

    def test_ca_cb_pair_drops_incomplete_residue(self):
        dims = (("CA", ("CA",)), ("CB", ("CB",)))
        pl = PeakList.from_chemshifts(_cs(_ROWS), dims=dims)
        # only residue 1 has both CA and CB
        self.assertEqual(list(pl.data["Seq_ID"]), [1])
        self.assertEqual(pl.data.iloc[0]["CB_ppm"], 19.0)

    def test_missing_dim_warns_and_returns_empty(self):
        dims = (("C", ("C",)), ("N", ("N",)))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pl = PeakList.from_chemshifts(_cs(_ROWS), dims=dims)
        self.assertTrue(any(issubclass(w.category, UserWarning) for w in caught))
        self.assertEqual(len(pl.data), 0)
        self.assertEqual(
            list(pl.data.columns),
            ["Seq_ID", "Auth_seq_ID", "Comp_ID", "C_ppm", "N_ppm", "assn_label"],
        )

    def test_no_overlap_warns_and_returns_empty(self):
        # H only on residue 1, a made-up "X" atom only on residue 2 -> no
        # residue has both, so the dims are individually present but never
        # share a row.
        rows = [
            {"ChemShift_ID": "sf1", "Entity_ID": 1, "Seq_ID": 1, "Auth_seq_ID": 1,
             "Comp_ID": "ALA", "Atom_ID": "H", "Val": 8.1},
            {"ChemShift_ID": "sf1", "Entity_ID": 1, "Seq_ID": 2, "Auth_seq_ID": 2,
             "Comp_ID": "GLY", "Atom_ID": "X", "Val": 1.0},
        ]
        dims = (("H", ("H",)), ("X", ("X",)))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pl = PeakList.from_chemshifts(_cs(rows), dims=dims)
        self.assertTrue(any(issubclass(w.category, UserWarning) for w in caught))
        self.assertEqual(len(pl.data), 0)


class TestFromCsv(unittest.TestCase):

    def _write(self, tmp_path, rows):
        df = pd.DataFrame(rows)
        path = tmp_path / "peaks.csv"
        df.to_csv(path, index=False)
        return str(path)

    def setUp(self):
        import tempfile
        from pathlib import Path
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_default_dims(self):
        rows = [
            {"res": "A1", "atom": "1H", "shift": 8.1},
            {"res": "A1", "atom": "15N", "shift": 120.0},
            {"res": "G2", "atom": "1H", "shift": 8.5},
            {"res": "G2", "atom": "15N", "shift": 110.0},
        ]
        path = self._write(self.tmp_path, rows)
        pl = PeakList.from_csv(path)
        self.assertEqual(
            list(pl.data.columns),
            ["Seq_ID", "Auth_seq_ID", "Comp_ID", "H_ppm", "N_ppm", "assn_label"],
        )
        self.assertEqual(len(pl.data), 2)
        self.assertEqual(pl.data.set_index("Seq_ID").loc[1, "H_ppm"], 8.1)

    def test_seq_offset(self):
        rows = [
            {"res": "A1", "atom": "1H", "shift": 8.1},
            {"res": "A1", "atom": "15N", "shift": 120.0},
        ]
        path = self._write(self.tmp_path, rows)
        pl = PeakList.from_csv(path, seq_offset=10)
        self.assertEqual(list(pl.data["Seq_ID"]), [11])

    def test_custom_dims(self):
        rows = [
            {"res": "A1", "atom": "13C", "shift": 52.0},
            {"res": "A1", "atom": "1H", "shift": 0.9},
        ]
        dims = (("C", ("13C",)), ("H", ("1H",)))
        path = self._write(self.tmp_path, rows)
        pl = PeakList.from_csv(path, dims=dims)
        self.assertEqual(
            list(pl.data.columns),
            ["Seq_ID", "Auth_seq_ID", "Comp_ID", "C_ppm", "H_ppm", "assn_label"],
        )
        self.assertEqual(pl.data.iloc[0]["C_ppm"], 52.0)

    def test_missing_dim_warns_and_returns_empty(self):
        rows = [
            {"res": "A1", "atom": "1H", "shift": 8.1},
        ]
        path = self._write(self.tmp_path, rows)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pl = PeakList.from_csv(path)
        self.assertTrue(any(issubclass(w.category, UserWarning) for w in caught))
        self.assertEqual(len(pl.data), 0)


class TestAssignmentString(unittest.TestCase):

    def test_with_custom_dims_peaklist(self):
        dims = (("CA", ("CA",)), ("CB", ("CB",)))
        pl = PeakList.from_chemshifts(_cs(_ROWS), dims=dims)
        # only Seq_ID 1 (ALA) made it through; sequence is A, G, A (pos 3 unassigned)
        s = pl.assignment_string(sequence="AGA")
        self.assertEqual(s, "A..")


if __name__ == "__main__":
    unittest.main()
