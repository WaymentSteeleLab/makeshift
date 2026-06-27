import unittest
import tempfile
import os
import makeshift as ms


BMRB_ID = 5363
ASSIGN_NAME = 'chemical_shifts_1'

EXPECTED_OFFSETS = {
    'N':  1.0884181598620282,
    'CA': -2.121216467191502,
    'CB': -2.1212146555686777,
    'C':  None,
    'H':  0.17819534303635887,
}

EXPECTED_CHECK = {'CA': True, 'CB': True, 'C': False, 'N': True, 'H': True}


def _fetch_and_run():
    with tempfile.TemporaryDirectory() as tmp:
        ms.fetch_nmrstar_file(BMRB_ID, output_dir=tmp)
        parsed = ms.parse_nmr_star(os.path.join(tmp, f'bmr{BMRB_ID}_3.str'))
        df = ms.get_chem_shifts(parsed)

    df_i = df[df['cs_saveframe_id'] == ASSIGN_NAME].copy()
    df_i['Seq_ID'] = df_i['Seq_ID'].astype(int)
    df_i = df_i.sort_values('Seq_ID').reset_index(drop=True)
    return ms.reref(df_i, method='lacs')


class TestRerefLacs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df, cls.check, cls.offsets = _fetch_and_run()

    def test_check_flags(self):
        self.assertEqual(self.check, EXPECTED_CHECK)

    def test_offsets_match(self):
        for atom, expected in EXPECTED_OFFSETS.items():
            with self.subTest(atom=atom):
                self.assertIn(atom, self.offsets)
                got = self.offsets[atom]
                if expected is None:
                    self.assertIsNone(got)
                else:
                    self.assertAlmostEqual(got, expected, places=6)

    def test_reref_mask_column_present(self):
        self.assertIn('reref_mask', self.df.columns)

    def test_orig_column_preserved(self):
        self.assertIn('orig', self.df.columns)
        self.assertFalse(self.df['orig'].equals(self.df['Val']))


if __name__ == '__main__':
    unittest.main()
