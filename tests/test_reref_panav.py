import unittest
import tempfile
import os
import makeshift as ms


BMRB_ID = 5363
ASSIGN_NAME = 'chemical_shifts_1'

# Round-2 (final) offsets from demo_panav.ipynb
EXPECTED_OFFSETS = {
    'N':  3.193450497798203e-16,
    'CA': -0.0020528219036572803,
    'CB': 0.009519163763066838,
    'C':  None,
}

EXPECTED_CHECK = {'N': True, 'CA': True, 'CB': True, 'C': False}


def _fetch_and_run():
    with tempfile.TemporaryDirectory() as tmp:
        ms.fetch_nmrstar_file(BMRB_ID, output_dir=tmp)
        parsed = ms.parse_nmr_star(os.path.join(tmp, f'bmr{BMRB_ID}_3.str'))
        df = ms.get_chem_shifts(parsed)

    df_i = df[df['cs_saveframe_id'] == ASSIGN_NAME].copy()
    df_i['Seq_ID'] = df_i['Seq_ID'].astype(int)
    df_i = df_i.sort_values('Seq_ID').reset_index(drop=True)
    return ms.reref(df_i, method='panav')


class TestRerefPanav(unittest.TestCase):

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

    def test_outlier_columns_present(self):
        self.assertIn('outlier_1', self.df.columns)
        self.assertIn('outlier_2', self.df.columns)

    def test_orig_column_preserved(self):
        self.assertIn('orig', self.df.columns)
        self.assertFalse(self.df['orig'].equals(self.df['Val']))


if __name__ == '__main__':
    unittest.main()
