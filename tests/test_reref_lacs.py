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
    'H':  0.17819534303635887,
}


def _fetch_and_run():
    with tempfile.TemporaryDirectory() as tmp:
        ms.fetch_nmrstar_file(BMRB_ID, output_dir=tmp)
        parsed = ms.parse_nmr_star(os.path.join(tmp, f'bmr{BMRB_ID}_3.str'))
        df = ms.get_chem_shifts(parsed)

    df_i = df[df['cs_saveframe_id'] == ASSIGN_NAME].copy()
    df_i['Seq_ID'] = df_i['Seq_ID'].astype(int)
    df_i = df_i.sort_values('Seq_ID').reset_index(drop=True)

    df_i = df_i.loc[
        df_i.Atom_ID.isin(['H', 'HA', 'N', 'CA', 'CB', 'C']) |
        df_i.Atom_ID.str.contains('^HA', na=False)
    ]

    # Average GLY HA/HA2, keep one as HA
    gly_df = df_i[(df_i['Comp_ID'] == 'GLY') & (df_i['Atom_ID'].str.contains('HA'))]
    for seq_id, group in gly_df.groupby('Seq_ID'):
        mean_val = group['Val'].mean()
        mask = (
            (df_i['Seq_ID'] == seq_id) &
            (df_i['Comp_ID'] == 'GLY') &
            df_i['Atom_ID'].str.contains('HA')
        )
        df_i.loc[mask, 'Val'] = mean_val
        indices = df_i.loc[mask].index
        df_i.loc[indices[0], 'Atom_ID'] = 'HA'
        if len(indices) > 1:
            df_i.loc[indices[1:], 'Atom_ID'] = 'HA2'

    df_i = df_i.loc[df_i.Atom_ID.isin(['H', 'HA', 'N', 'CA', 'CB', 'C'])].copy()
    return ms.reref_lacs_(df_i)


class TestRerefLacs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df, cls.skip, cls.offsets = _fetch_and_run()

    def test_not_skipped(self):
        self.assertFalse(self.skip)

    def test_offsets_match(self):
        for atom, expected in EXPECTED_OFFSETS.items():
            with self.subTest(atom=atom):
                self.assertIn(atom, self.offsets)
                self.assertAlmostEqual(self.offsets[atom], expected, places=6)

    def test_c_not_fitted(self):
        self.assertNotIn('C', self.offsets)

    def test_reref_mask_column_present(self):
        self.assertIn('reref_mask', self.df.columns)

    def test_orig_column_preserved(self):
        self.assertIn('orig', self.df.columns)
        self.assertFalse(self.df['orig'].equals(self.df['Val']))


if __name__ == '__main__':
    unittest.main()
