import pandas as pd


def pivot_chem_shifts(cs, atoms=None):
    """Pivot long-format chem shifts DataFrame to wide format for plotting.

    Parameters
    ----------
    cs : DataFrame
        Output of get_chem_shifts() with columns Seq_ID, Comp_ID, Atom_ID, Val.
    atoms : list of str, optional
        Subset of atom names to include (e.g. ['H', 'N']). All atoms if None.

    Returns
    -------
    DataFrame with one row per residue and one column per atom type.
    """
    if atoms is not None:
        cs = cs[cs['Atom_ID'].isin(atoms)]
    wide = cs.pivot_table(
        index=['Seq_ID', 'Comp_ID'],
        columns='Atom_ID',
        values='Val',
    ).reset_index()
    wide.columns.name = None
    return wide
