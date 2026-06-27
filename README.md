# `makeshift`: lightweight NMR tools

A dependency-light Python toolkit for working with protein NMR data: parsing
[NMR-STAR](https://bmrb.io/spec/) files from the [BMRB](https://bmrb.io/),
re-referencing chemical shifts, building peak lists, and running a full CPMG
relaxation-dispersion pipeline.

The core (BMRB parsing, chemical shifts, re-referencing) needs only `numpy`,
`pandas`, `scipy`, and `scikit-learn`. The heavier spectrum and relaxation
tools are opt-in, so `import makeshift` stays light.

## Installation

```bash
pip install git+https://github.com/gelnesr/makeshift.git
```

Editable install (for development):

```bash
git clone https://github.com/gelnesr/makeshift.git
cd makeshift
pip install -e .
```

The demo notebooks need a few extras:

```bash
pip install -e ".[demos]"          # seaborn, matplotlib, jupyter
```

The spectrum/CPMG pipeline (`makeshift.spectra`, `makeshift.relaxation`)
additionally needs `nmrglue`, `tqdm`, `matplotlib`, and `seaborn`.

## Quickstart

```python
import makeshift as ms

# Fetch and parse a BMRB entry into tidy chemical shifts
cs = ms.ChemicalShifts.from_bmrb(5363)
cs.data            # one row per shift: Seq_ID, Comp_ID, Atom_ID, Atom_type, Val
cs.sequences()     # one row per entity: ID, polymer type, sequence

# Re-reference, then compute the Chemical Shift Index
cs = ms.ChemicalShifts.from_bmrb(4527, reref="lacs", calc_csi=True)
cs.reref_offsets   # {atom: offset applied}

# Build an assigned peak list (e.g. for an HSQC)
peaks = cs.peaklist()
peaks.data
```

## Modules

| Module | What it does |
|---|---|
| `makeshift` (core) | `ChemicalShifts`, `NMRStarEntry`, `PeakList` — fetch/parse BMRB entries, extract shifts and sequences, build peak lists. |
| `makeshift.reref` | LACS and PANAV chemical-shift re-referencing (via `ChemicalShifts.reref`). |
| `makeshift.spectra` | Read Sparky `.ucsf` spectra (`Spectrum`), pick peaks, and align peak lists (`map_peaklists`). |
| `makeshift.relaxation` | CPMG relaxation-dispersion pipeline (`CPMGExperiment`): planes → R₂,eff → per-residue classification. |
| `makeshift.hydronmr` | Predict per-residue T1/T2/NOE from a PDB structure (`run`). |

See `demos/` for worked examples: `quick_start.ipynb` (core workflow),
`reref.ipynb` (re-referencing), and `cpmg_demo.ipynb` (the CPMG pipeline).

## Re-referencing

BMRB shifts are sometimes mis-referenced — a constant offset shifts every peak
of a given nucleus. `ChemicalShifts.reref` corrects this in place using one of
two methods:

- **PANAV** ([Wang & Wishart 2005](https://pubmed.ncbi.nlm.nih.gov/15772753/)) —
  uses rarely-misreferenced HA shifts to assign secondary structure, then aligns
  N/CA/CB to curated per-structure reference distributions
  ([Wang & Jardetzky 2002](https://onlinelibrary.wiley.com/doi/10.1110/ps.3180102)).
- **LACS** ([Wang & Markley 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2782637/)) —
  fits secondary shift vs. CSI so the random-coil regime intercepts at the origin;
  covers CA, CB, C′, N, and HN.

```python
cs = ms.ChemicalShifts.from_bmrb(4527)
cs.reref(method="panav")   # or "lacs"
print(cs.reref_offsets)    # {'N': ..., 'CA': ..., 'CB': ..., ...}
```

![Re-referencing example](static/example_rereferencing_ed.png)

Entry 4527 is correctly referenced; entries 6586 and 4150 have been described in
the literature as needing re-referencing. The two methods have not yet been
extensively compared.

## NMR-STAR concepts

NMR-STAR files are organised around **saveframes**, each belonging to a category
(e.g. `assigned_chemical_shifts`, `entity`, `sample`). The three you interact
with most:

- **Entry** — a single BMRB deposition (one `.str` file).
- **Entity** — a distinct molecular species (protein, DNA strand, ligand), each
  with its own `Entity_ID`.
- **Chemical shift list** — the `_Atom_chem_shift` loop inside an
  `assigned_chemical_shifts` saveframe; one row per observed shift.

## License

MIT License.

## Acknowledgments

- [BMRB](https://bmrb.io/) for maintaining and sharing NMR data.