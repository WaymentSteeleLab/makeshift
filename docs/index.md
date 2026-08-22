# makeshift

An open-source Python package for accessing and analyzing NMR data, from either custom input or
[NMR-STAR](https://pynmrstar.readthedocs.io/en/latest/) files from the
[BMRB](https://bmrb.io/).

Implementation details and validation of makeshift are described in our preprint: El Nesr & Wayment-Steele, [*Makeshift: a lightweight software for accessing and analyzing NMR data and protein dynamics*](https://www.biorxiv.org/content/10.64898/2026.08.17.745346v1) (bioRxiv, 2026). [doi:10.64898/2026.08.17.745346](https://doi.org/10.64898/2026.08.17.745346).

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
|-----------|---|
| [`makeshift`](api/entry.md) (core) | Fetch/parse BMRB entries; extract chemical shifts, sequences, relaxation and order-parameter data; build assigned peak lists. Classes: [`NMRStarEntry`](api/entry.md), [`ChemicalShifts`](api/chemshift.md), [`PeakList`](api/peaklist.md). |
| [`makeshift.reref`](api/reref.md) | LACS and PANAV chemical-shift re-referencing (via `ChemicalShifts.reref`). |
| [`makeshift.spectra`](api/spectra.md) | Read Sparky `.ucsf` spectra ([`Spectrum`](api/spectra.md)), pick peaks, and align peak lists (`map_peaklists`). |
| [`makeshift.relaxation`](api/relaxation.md) | CPMG dispersion pipeline ([`CPMGExperiment`](api/cpmg.md)) and [`RelaxationProfile`](api/relaxation.md) — RelaxDB-style per-residue dynamics from deposited R1/R2/NOE. |
| [`makeshift.hydronmr`](api/hydronmr.md) | Predict per-residue T1/T2/NOE from a PDB structure. |
| [`makeshift.talosn`](api/talosn.md) | Predict backbone torsion angles, S² order parameters, and secondary structure from chemical shifts via the NIH TALOS-N binary. |
| [`makeshift.rci`](api/rci.md) | Predict per-residue backbone flexibility (Random Coil Index) from chemical shifts, in pure Python. |
| [`makeshift.utils`](api/utils.md) | Dependency-light helpers: dataset/structure fetching, constants. |

## Where to go next

<div class="grid cards" markdown>

- :material-download: **[Installation](installation.md)** — install the package and optional extras.
- :material-rocket-launch: **[Quickstart](quickstart.md)** — the core fetch → shifts → peaks workflow.
- :material-book-open-variant: **[User guide](guide/entries.md)** — task-focused walkthroughs of every module.
- :material-api: **[API reference](api/index.md)** — full signatures and docstrings, generated from the source.
- :material-book-open-page-variant: **[Citation](citation.md)** — how to cite makeshift and related work.

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

## Citation

If you use makeshift, please cite:

```bibtex
@article{makeshift2026,
  title   = {makeshift: a lightweight software for accessing and analyzing NMR data and protein dynamics},
  author  = {El Nesr, Gina and Wayment-Steele, Hannah K.},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.08.17.745346},
  url     = {https://doi.org/10.64898/2026.08.17.745346}
}
```

If you use the relaxation-dispersion processing, please also cite:

```bibtex
@article {dyna1,
    author = {Wayment-Steele, Hannah K. and El Nesr, Gina and Hettiarachchi, Ramith and Ojoawo, Adedolapo and Kariyawasam, Hasindu and Ovchinnikov, Sergey and Kern, Dorothee},
    title = {Learning millisecond protein dynamics from what is missing in NMR spectra},
    year = {2026},
    doi = {10.1038/s41586-026-10989-4},
    journal = {Nature}
}
```