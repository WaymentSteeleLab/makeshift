import re, os
from collections import defaultdict
import pandas as pd
import numpy as np
import urllib.request

def fetch_nmrstar_file(bmrb_id):
    path=f"https://bmrb.io/ftp/pub/bmrb/entry_directories/bmr{bmrb_id}/bmr{bmrb_id}_3.str"
    return urllib.request.urlretrieve(path, f"bmr{bmrb_id}_3.str")

def parse_nmr_star(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data = {}
    saveframe_name = None
    in_saveframe, in_loop = False, False
    loop_tags, loop_data = [], []
    current_saveframe, current_tags = {},{}
    current_loops = defaultdict(list)

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('save_'):
            if in_saveframe:
                current_saveframe.update(current_tags)
                current_saveframe.update(dict(current_loops))
                if saveframe_name:
                    data[saveframe_name] = current_saveframe #

            saveframe_name = line[5:].strip() or None
            current_saveframe = {}
            current_tags = {}
            current_loops = defaultdict(list)
            loop_tags = []
            loop_data = []
            in_loop = False
            in_saveframe = bool(saveframe_name)
            i += 1
            continue

        if in_saveframe:
            if line == 'loop_':
                in_loop = True
                loop_tags, loop_data = [], []
                i += 1
                while i < len(lines):
                    tag_line = lines[i].strip()
                    if tag_line.startswith('_'):
                        loop_tags.append(tag_line)
                        i += 1
                    else:
                        break
                while i < len(lines):
                    data_line = lines[i].strip()
                    if data_line == 'stop_':
                        break
                    if data_line:
                        values = re.findall(r'(?:"[^"]*"|\'[^\']*\'|[^\s]+)', data_line)
                        if len(values) == len(loop_tags):
                            base_prefix = loop_tags[0].split('.')[0] + '.'
                            row = {
                                tag[len(base_prefix):] if tag.startswith(base_prefix) else tag: val.strip('"')
                                for tag, val in zip(loop_tags, values)
                            }
                            loop_category = base_prefix.rstrip('.')
                            current_loops[loop_category].append(row)
                    i += 1
                in_loop = False
                i += 1
                continue

            elif line == 'stop_':
                current_saveframe.update(current_tags)
                current_saveframe.update(dict(current_loops))
                if saveframe_name:
                    data[saveframe_name] = current_saveframe #

                current_saveframe, current_tags = {}, {}
                current_loops = defaultdict(list)
                loop_tags, loop_data = [], []
                in_loop, in_saveframe = False, False
                saveframe_name = None
                i += 1
                continue

            elif line.startswith('_'):
                tag = line.split()[0]
                key = tag.split('.')[-1] if '.' in tag else tag
                # Check for multiline value
                if len(line.split()) == 1 and i + 1 < len(lines) and lines[i + 1].strip() == ';':
                    i += 2
                    value_lines = []
                    while i < len(lines):
                        val_line = lines[i].rstrip('\n')
                        if val_line.strip() == ';':
                            break
                        value_lines.append(val_line)
                        i += 1
                    value = ''.join(value_lines)
                    current_tags[key] = value
                    i += 1  # move past closing ';'
                else:
                    parts = line.split(None, 1)
                    value = parts[1] if len(parts) > 1 else ''
                    current_tags[key] = value.strip('"\'')
                    i += 1
                continue
            else:
                i += 1
                continue
        else:
            i += 1
            continue

    if in_saveframe and saveframe_name:
        current_saveframe.update(current_tags)
        current_saveframe.update(dict(current_loops))
        data[saveframe_name] = current_saveframe #

    # restructure

    data_2 = {}

    for k, v in data.items():
      Sf_category = v['Sf_category']
      Sf_framecode = v['Sf_framecode']
      if Sf_category not in data_2.keys():
        data_2[Sf_category] = {}

      data_2[Sf_category][Sf_framecode] = v
        
    return data_2

def convert_loop_to_dataframe(loop):
    dct = {k:[] for k in loop[0].keys()}
    for entr in loop:
        for k, v in entr.items():
            dct[k].append(v)
    return pd.DataFrame.from_records(dct)

def clean_cs_dataframe(df):
    df['Val'] = df['Val'].replace('.',np.nan)
    df['Val'] = df['Val'].astype(float)
    
    return df[['Entity_ID', 'Seq_ID', 'Auth_seq_ID','Comp_ID',
                'Atom_ID','Atom_type','Val']]

def get_sequences(parsed):
    outs=[]
    tags=['ID', 'Polymer_type', 'Polymer_seq_one_letter_code']
    for k, entry in parsed['entity'].items():
      out = {'entity': k}
      out.update({i: entry[i] for i in tags})
      outs.append(out)

    return pd.DataFrame.from_records(outs)

def get_sample_info(parsed):
    outs = []
    tags=['ID', 'Mol_common_name','Entity_ID','Isotopic_labeling','Concentration_val','Concentration_val_units']

    for k, entry in parsed['sample'].items():
      for j, x in enumerate(entry['_Sample_component']):
        out = {'sample': k}
        out.update({i: x[i] for i in tags})
        outs.append(out)

    return pd.DataFrame.from_records(outs)

def get_chem_shifts(parsed, calc_CSI=False):

    out_dfs=[]
    for k, entry in parsed['assigned_chemical_shifts'].items():
      cs = convert_loop_to_dataframe(entry['_Atom_chem_shift'])
      cs = clean_cs_dataframe(cs)
      try:
          cs['name'] = parsed[k]['Name']
      except:
          cs['name'] = '.'

      cs['cs_saveframe_id'] = k
      out_dfs.append(cs)
    out = pd.concat(out_dfs)
    out = out.reset_index()
    out['Seq_ID'] = out['Seq_ID'].astype(int)
    
    if calc_CSI:
        if 'CA' and 'CB' not in out['Atom_ID'].unique():
            raise RuntimeError('Cannot find CA and CB in Atom_ID, cannot calc CSI')
            
        def get_secondary_shift(row):
            """Calculate secondary shift (observed - random coil)"""
            return row['Val'] - RANDOM_COIL[row['Comp_ID']][row['Atom_ID']]
        
        def get_csi(row, df):
            data = df.loc[df.Seq_ID==row['Seq_ID']]
            ca = data[data['Atom_ID'] == 'CA']
            cb = data[data['Atom_ID'] == 'CB']
            if len(ca) >= 1 and len(cb) >= 1:
                ca_sec = get_secondary_shift(ca.iloc[0])
                cb_sec = get_secondary_shift(cb.iloc[0])

                if not (np.isnan(ca_sec) or np.isnan(cb_sec)):
                    return ca_sec - cb_sec
                else:
                    return np.nan
            else:
                return np.nan
            
        out['csi'] = out.apply(lambda row: get_csi(row, out), axis=1)
        
    return out

RANDOM_COIL = {
    'ALA': {'CA': 52.5, 'CB': 19.1, 'N': 123.8, 'H': 8.24},
    'ARG': {'CA': 56.0, 'CB': 30.2, 'N': 120.5, 'H': 8.27},
    'ASN': {'CA': 53.1, 'CB': 38.9, 'N': 118.7, 'H': 8.40},
    'ASP': {'CA': 54.2, 'CB': 41.1, 'N': 120.4, 'H': 8.34},
    'CYS': {'CA': 58.2, 'CB': 28.0, 'N': 118.6, 'H': 8.31},
    'GLU': {'CA': 56.6, 'CB': 29.9, 'N': 120.2, 'H': 8.37},
    'GLN': {'CA': 55.7, 'CB': 29.4, 'N': 119.8, 'H': 8.32},
    'GLY': {'CA': 45.1, 'CB': np.nan, 'N': 108.8, 'H': 8.33},
    'HIS': {'CA': 55.0, 'CB': 29.9, 'N': 118.2, 'H': 8.42},
    'ILE': {'CA': 61.1, 'CB': 38.8, 'N': 119.9, 'H': 8.00},
    'LEU': {'CA': 55.1, 'CB': 42.4, 'N': 121.8, 'H': 8.16},
    'LYS': {'CA': 56.2, 'CB': 32.7, 'N': 120.4, 'H': 8.29},
    'MET': {'CA': 55.4, 'CB': 32.9, 'N': 119.6, 'H': 8.28},
    'PHE': {'CA': 57.7, 'CB': 39.6, 'N': 120.3, 'H': 8.30},
    'PRO': {'CA': 63.3, 'CB': 31.7, 'N': np.nan, 'H': np.nan},
    'SER': {'CA': 58.3, 'CB': 63.8, 'N': 115.7, 'H': 8.31},
    'THR': {'CA': 61.8, 'CB': 69.6, 'N': 113.6, 'H': 8.15},
    'TRP': {'CA': 57.5, 'CB': 29.6, 'N': 121.3, 'H': 8.25},
    'TYR': {'CA': 57.9, 'CB': 38.8, 'N': 120.3, 'H': 8.12},
    'VAL': {'CA': 62.2, 'CB': 32.9, 'N': 119.2, 'H': 8.03}
}

def get_assembly_info(parsed):
    outs = []
    tags=['Entity_assembly_name']

    for k, entry in parsed['assembly'].items():
      for j, x in enumerate(entry['_Entity_assembly']):
        out = {'sample': k}
        out.update({i: x[i] for i in tags})
        outs.append(out)

    return pd.DataFrame.from_records(outs)
