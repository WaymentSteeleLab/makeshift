# makeshift

**Lightweight NMR tools** — a dependency-light open-source Python package for working with
biomolecular NMR data, from either custom input or
[NMR-STAR](https://pynmrstar.readthedocs.io/en/latest/) files from the
[BMRB](https://bmrb.io/).

```python
import makeshift as ms

# Fetch and parse a BMRB entry into tidy chemical shifts
cs = ms.ChemicalShifts.from_bmrb(5363)
cs.data            # one row per shift: Seq_ID, Comp_ID, Atom_ID, Atom_type, Val
cs.sequences()     # one row per entity: ID, polymer type, sequence
```

## What it does

`makeshift` turns deposited NMR data into tidy, analysis-ready tables and runs a
handful of common downstream analyses without pulling in a heavyweight
dependency stack.

| Module | What it does |
|-------|---|
| [`makeshift`](api/entry.md) (core) | Fetch/parse BMRB entries; extract chemical shifts, sequences, relaxation and order-parameter data; build assigned peak lists. Classes: [`NMRStarEntry`](api/entry.md), [`ChemicalShifts`](api/chemshift.md), [`PeakList`](api/peaklist.md). |
| [`makeshift.reref`](api/reref.md) | LACS and PANAV chemical-shift re-referencing (via `ChemicalShifts.reref`). |
| [`makeshift.spectra`](api/spectra.md) | Read Sparky `.ucsf` spectra ([`Spectrum`](api/spectra.md)), pick peaks, and align peak lists (`map_peaklists`). |
| [`makeshift.relaxation`](api/relaxation.md) | CPMG dispersion pipeline ([`CPMGExperiment`](api/cpmg.md)) and [`RelaxationProfile`](api/relaxation.md) — RelaxDB-style per-residue dynamics from deposited R1/R2/NOE. |
| [`makeshift.hydronmr`](api/hydronmr.md) | Predict per-residue T1/T2/NOE from a PDB structure. |
| [`makeshift.talosn`](api/talosn.md) | Predict backbone torsion angles, S² order parameters, and secondary structure from chemical shifts via the NIH TALOS-N binary. |
| [`makeshift.utils`](api/utils.md) | Dependency-light helpers: dataset/structure fetching, constants. |

## Where to go next

<div class="grid cards" markdown>

- :material-download: **[Installation](installation.md)** — install the package and optional extras.
- :material-rocket-launch: **[Quickstart](quickstart.md)** — the core fetch → shifts → peaks workflow.
- :material-book-open-variant: **[User guide](guide/entries.md)** — task-focused walkthroughs of every module.
- :material-api: **[API reference](api/index.md)** — full signatures and docstrings, generated from the source.

</div>

## License

MIT License. 

Note that `makeshift.talosn` downloads and runs the TALOS-N binary,
which is distributed separately by NIH under its own
[Terms of Use](https://spin.niddk.nih.gov/bax-apps/terms.html); those terms
govern the downloaded software, not this wrapper.

## Acknowledgments

- The [Biological Magnetic Resonance Bank (BMRB)](https://bmrb.io/) for maintaining and sharing NMR data.
- The Bax lab at NIH for [TALOS-N](https://spin.niddk.nih.gov/bax-apps/software/TALOS-N/).
