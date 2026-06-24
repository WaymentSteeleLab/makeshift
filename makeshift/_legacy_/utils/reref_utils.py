def apply_offset(row, offsets):
    if row['Atom_ID'] in offsets.keys():
        return row['Val'] - offsets[row['Atom_ID']]
    else:
        return row['Val']
