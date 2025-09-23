import pandas as pd

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

def get_assembly_info(parsed):
    outs = []
    tags=['Entity_assembly_name']

    for k, entry in parsed['assembly'].items():
      for j, x in enumerate(entry['_Entity_assembly']):
        out = {'sample': k}
        out.update({i: x[i] for i in tags})
        outs.append(out)

    return pd.DataFrame.from_records(outs)