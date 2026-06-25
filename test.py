import re
import makeshift as ms
import pandas as pd

df = pd.read_csv('data_hsqcs_24jun26_metadata.csv')

import tqdm 
for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
    entry_id   = row['BMRB_entry']
    entity_id  = row['Entity_ID']
    chem_shift = row['chem_shift_id']
    cs  = ms.ChemicalShifts.from_bmrb(int(entry_id), keep_download=False)
    seq = cs.sequences(entity_id=entity_id)
    pl  = cs.peaklist(cs_saveframe=chem_shift, entity_id=entity_id)
    assigned_str = pl.assignment_string()
