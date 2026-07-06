# Reading BMRB entries

[`NMRStarEntry`](../api/entry.md) is the entry point for everything deposited in
an NMR-STAR file: sequences, samples, spectrometers, referencing, relaxation
data, order parameters, and citation metadata. If a concept here is unfamiliar,
see [NMR-STAR concepts](../concepts.md) first.

## Loading an entry

```python
import makeshift as ms

# Download from the BMRB by accession number
entry = ms.NMRStarEntry.from_bmrb(25013)

# ...or parse a local .str file
entry = ms.NMRStarEntry.from_file("bmr25013.str")
```

`from_bmrb` downloads to a temporary location by default; pass
`output_dir=` and `keep_download=True` to retain the file.

## Discovering what's inside

```python
entry.datasets()          # one row per data type present, with counts
entry.categories()        # attribute-accessible map of saveframe categories
entry.saveframe("entity") # a specific saveframe (or all framecodes in a category)
```

`datasets()` is the fastest way to see whether an entry carries chemical shifts,
relaxation data, order parameters, and so on before you dig in.

## Sequences and entities

```python
entry.sequences()          # ID, polymer type, one-letter sequence per entity
entry.polymer_type()       # polymer type per entity
entry.sequences()          # or entry.sequences(entity_id=1)
```

## Sample and acquisition metadata

```python
entry.sample_info()        # one row per sample component
entry.sample_conditions()  # pH, temperature, etc. per condition set
entry.assembly_info()      # one row per entity assembly
entry.spectrometers()      # name, manufacturer, model, field strength
entry.shift_reference()    # how the depositor referenced each nucleus
```

Several boolean helpers summarise labelling and sample state. Called with an id
they return a bool; called bare they return a table across all samples/entities:

```python
entry.is_deuterated()       # or is_deuterated(entity_id=1)
entry.is_methyl_labeled()
entry.is_denatured()
```

## Relaxation and order parameters

```python
entry.relaxation("T2")      # R2 — also "T1"/"R1", "T1rho", "NOE"; units-aware
entry.order_parameters()    # model-free S2 (S2, Tau_e, Rex)
```

These return tidy per-residue DataFrames. To go further — assembling R1/R2/NOE
into a motional profile — see [Relaxation & dynamics](relaxation.md).

## Anything else

For loops without a dedicated method, use the generic escape hatch:

```python
entry.data_loop("spectral_density_values", "_Spectral_density")
```

## Cross-references and citation

```python
entry.get_pdb_ids()          # cited PDB ids
entry.get_alphafold_ids()    # cited AlphaFold / UniProt accessions
entry.db_links()             # database cross-references
entry.get_entry_title()
entry.citation_info()        # title, journal, year, DOI, PubMed, authors
```

The PDB/AlphaFold links feed directly into structure-based predictions like
[HYDRONMR](hydronmr.md) and the rigid-body step of a
[relaxation profile](relaxation.md).

## Full API

See the [`NMRStarEntry` reference](../api/entry.md) for every method and argument.
