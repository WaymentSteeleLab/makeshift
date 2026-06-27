# `makeshift`: Lightweight NMR tools

This repository provides a minimal, dependency-light tools to handle NMR data. Makeshift reads [NMR-STAR](https://bmrb.io/spec/) format files from the Biological Magnetic Resonance Bank ([BMRB](https://bmrb.io/)). It extracts sample metadata and measurements into Python dictionaries and pandas DataFrames.

## Features

- Parse `.str` files into nested Python dictionaries
- Re-reference, calculate CSI, and more in `python`
- Extract polymer sequences, sample compositions, and chemical shifts
- Built using Python standard library + `pandas`

## Installation

```bash
pip install git+https://github.com/WaymentSteeleLab/makeshift.git
```

Or clone and install in editable mode (useful if you want to modify the code):

```bash
git clone https://github.com/WaymentSteeleLab/makeshift.git
cd makeshift
pip install -e .
```

To also install dependencies for running the demo notebooks (seaborn, matplotlib, jupyter):

```bash
pip install -e ".[demos]"
```

---

## Quickstart

```python
import makeshift as ms

# Download an NMR-STAR file from BMRB
ms.fetch_nmrstar_file(5363)  # saves as bmr5363_3.str

# Parse the file
entry = ms.parse_nmr_star('bmr5363_3.str')

# Extract useful information
seq = ms.get_sequences(entry)
samples = ms.get_sample_info(entry)
cs = ms.get_chem_shifts(entry)
```
---

## Probabilistic Re-Referencing

Two lightweight re-referencing implementations are available:

`panav`: This code implements the core idea described in [Wang & Wishart 2005](https://pubmed.ncbi.nlm.nih.gov/15772753/). The idea is to use HA atoms, hydrogens being rarely mis-referenced, to estimate secondary structure from curated shift distributions from [Wang & Jardetsky 2002](https://onlinelibrary.wiley.com/doi/10.1110/ps.3180102). The method then minimizes the difference in distribution for N, CA, CB atoms between the current distribution and the curated shift distribution.

`lacs`: This code implements the core idea described in [Wang & Markley 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2782637/). The idea is to enforce that the chemical shift index (CSI) of the i-1 Carbon and the i Nitrogen intercepts at (0,0), essentially setting the "random coil" regime of each protein to be there.

Note: these two methods have not yet been extensively compared.

```python
ms.fetch_nmrstar_file(4527)
cs = ms.get_chem_shifts(ms.parse_nmr_star('bmr4527_3.str'))

df, check, offsets = ms.reref(cs, method='panav')  # or method='lacs'
print(offsets)  # {'N': ..., 'CA': ..., 'CB': ..., 'C': ...}
```
![Image showing distributions](https://github.com/WaymentSteeleLab/NMRstar_parser/blob/c9726af327b8dc4fef08c0c25711448342b0fe7f/static/example_rereferencing_ed.png)

Entry 4527 is an example entry that is correctly referenced. Entries 6586 and 4150 are both entries described previously in literature as needing re-referencing.

---

## NMR-STAR concepts

NMR-STAR files are organised around **saveframes**. Each saveframe belongs to a category (e.g. `assigned_chemical_shifts`, `entity`, `sample`) and contains key-value pairs plus data loops.

The three concepts you’ll interact with most:

**Entry** — a single BMRB deposition. One `.str` file, one entry.

**Entity** — a distinct molecular species (protein, DNA strand, ligand, etc.). Multi-component complexes have multiple entities, each with its own `Entity_ID`.

**Chemical shift list** — the `_Atom_chem_shift` loop inside an `assigned_chemical_shifts` saveframe. One row per observed shift, keyed by `Entity_ID`, `Seq_ID`, `Comp_ID`, and `Atom_ID`.

---

## API

### Fetching and parsing

| Function | Description |
|---|---|
| `fetch_nmrstar_file(bmrb_id)` | Download the NMR-STAR v3 file for a BMRB entry and save it locally. |
| `parse_nmr_star(file_path)` | Parse a `.str` file into a nested dict keyed by saveframe category. |

### Extracting data

| Function | Description |
|---|---|
| `get_sequences(parsed)` | DataFrame of entities: ID, polymer type, one-letter sequence. |
| `get_sample_info(parsed)` | DataFrame of sample components: name, labeling, concentration. |
| `get_chem_shifts(parsed, calc_CSI=False)` | Tidy DataFrame of assigned shifts — one row per observation. Pass `calc_CSI=True` to add `csi_raw` and `csi` columns. |

### Re-referencing

| Function | Description |
|---|---|
| `reref(df, method)` | Correct backbone referencing errors. `method` is `’panav’` or `’lacs’`. Returns `(df, check, offsets)`. |

`reref` return values:
- `df` — corrected shifts; `orig` column holds the pre-correction values
- `check` — `{atom: bool}` indicating which atom types converged
- `offsets` — `{atom: float | None}` total offset applied per atom type

---

## License

MIT License

---

## Acknowledgments

- [BMRB](https://bmrb.io/) for maintaining and sharing NMR data.
