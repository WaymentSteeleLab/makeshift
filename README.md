# `nmr_parsing`: Lightweight NMR tools

This repository provides a minimal, dependency-light tools to handle NMR data. nmr_star_parser reads [NMR-STAR](https://bmrb.io/spec/) format files from the Biological Magnetic Resonance Bank ([BMRB](https://bmrb.io/)). It extracts sample metadata and measurements into Python dictionaries and pandas DataFrames.

## Features

- Parse `.str` files into nested Python dictionaries
- Extract polymer sequences, sample compositions, and chemical shifts
- Built using Python standard library + `pandas`

## Installation

Clone this repo or copy `parser.py` into your project.

---

## Quickstart

See in Google colab [demo!](https://github.com/hkws-lab/NMRstar_parser/blob/main/nmrstar_parser_demo.ipynb)

```python
from parser import *

# Download an NMR-STAR file from BMRB
fetch_nmrstar_file(5363)  # saves as bmr5363_3.str

# Parse the file
entry = parse_nmr_star('bmr5363_3.str')

# Extract useful information
seq = get_sequences(entry)
samples = get_sample_info(entry)
cs = get_chem_shifts(entry)
```
---

## Probabilistic Re-Referencing

This code implements the core idea described in [Wang & Wishart 2005](https://pubmed.ncbi.nlm.nih.gov/15772753/). The idea is to use HA atoms, hydrogens being rarely mis-referenced, to estimate secondary structure from curated shift distributions from [Wang & Jardetsky 2002](https://onlinelibrary.wiley.com/doi/10.1110/ps.3180102). The method then minimizes the difference in distribution for N, CA, CB atoms between the current distribution and the curated shift distribution.

```python
%pylab inline
from ReRef import *
inds=[4527,6586,4150]

figure(figsize=(10,2))
colors = sns.color_palette()

for j, ind in enumerate(inds):

    fetch_nmrstar_file(ind) # if nmrstar file from bmrb is not downloaded

    df = get_chem_shifts(parse_nmr_star(f'bmr{ind}_3.str'))
    df = df.loc[df.Atom_ID.isin(['H','HA','N','CA','CB'])]
    df = rereference(df)

    #compare distributions before and after
    for i,atom_id in enumerate(['HA','H','N','CA','CB']):
        subplot(1,5,i+1)
        sns.kdeplot(df.loc[df.Atom_ID==atom_id]['Val'],color=colors[j],label=ind)
        sns.kdeplot(df.loc[df.Atom_ID==atom_id]['orig'],color=colors[j],linestyle=':')
        xlabel(f'omega {atom_id[0]} (ppm)')
        title(atom_id)
        if i==0:
            legend()
tight_layout()
```
![Image showing distributions](https://github.com/WaymentSteeleLab/NMRstar_parser/blob/c9726af327b8dc4fef08c0c25711448342b0fe7f/static/example_rereferencing_ed.png)

Entry 4527 is an example entry that is correctly referenced. Entries 6586 and 4150 are both entries described previously in literature as needing re-referencing.

---

## 🔎 Crash course on BMRB Attributes

To make sense of the structured data in NMR-STAR files, here are key categories you’ll encounter when parsing BMRB entries:

### 🗂️ Entry
The top-level object in a BMRB deposition. Each `.str` file represents one **entry**, corresponding to a single experiment or dataset deposited to the BMRB.

- Typically contains many **saveframes** describing samples, sequences, chemical shifts, and experimental conditions.

### 🔬 Entity
An **entity** refers to a distinct molecular species (e.g. a polypeptide, DNA, RNA, ligand, etc.) in the sample.

- Attributes often include:
  - `Polymer_type`: e.g. `polypeptide(L)`
  - `Polymer_seq_one_letter_code`: the primary sequence of the entity
  - `ID`: a numeric identifier used to cross-reference with samples and assignments

### 🧪 Sample
A **sample** is the physical material used in an NMR experiment, typically consisting of one or more entities in solution.

- The `sample` saveframe includes:
  - `Mol_common_name`: e.g. "Ubiquitin"
  - `Isotopic_labeling`: labeling strategy (e.g., `15N`, `13C`)
  - `Concentration_val`: numeric concentration
  - `_Sample_component`: a loop describing the components and their entity associations

### 📈 Chemical Shift List
This is typically found in `assigned_chemical_shifts` saveframes and includes experimental chemical shift data.

- The `_Atom_chem_shift` loop contains:
  - `Entity_ID`, `Seq_ID`, `Comp_ID`, `Atom_ID`
  - `Val`: chemical shift value (in ppm)
  - `Val_err`: uncertainty (in ppm)
  - Often accompanied by a `Name` for the dataset

---


## API Overview

### `fetch_nmrstar_file(bmrb_id)`
Downloads the version 3 NMR-STAR `.str` file from BMRB given a numeric ID.

### `parse_nmr_star(file_path)`
Parses the `.str` file into a nested dictionary organized by saveframe.

### `get_sequences(parsed)`
Returns a DataFrame summarizing polymer types and sequences from the parsed entry.

### `get_sample_info(parsed)`
Returns sample metadata including labeling and concentration info as a DataFrame.

### `get_chem_shifts(parsed)`
Extracts assigned chemical shifts into a DataFrame, including atom type, chemical shift, and associated metadata.

---

## Example Output

### `get_sequences`
| ID | Polymer_type | Polymer_seq_one_letter_code |
|----|---------------|-----------------------------|
| 1  | polypeptide(L) | MKTAYIAKQRQISFVKSHFSRQLE... |

### `get_sample_info`
| ID | Mol_common_name | Isotopic_labeling | Concentration_val | ... |
|----|------------------|--------------------|-------------------|-----|
| 1  | Ubiquitin        | 15N,13C            | 1.0               |     |

### `get_chem_shifts`
| Entity_ID | Seq_ID | Atom_ID | Val  | Val_err | name | cs_saveframe_id |
|-----------|--------|---------|------|---------|------|------------------|
| 1         | 23     | HN      | 8.23 | 0.01    | .    | sf_chemshift_1   |

---

## License

MIT License

---

## Acknowledgments

- [BMRB](https://bmrb.io/) for maintaining and sharing NMR data
