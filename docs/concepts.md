# Core concepts

`makeshift` centres on three objects: `NMRStarEntry`, `ChemicalShifts`, and
`PeakList`. Understand how they connect and what each does, and the rest of the
API follows naturally.

## NMRStarEntry

[`NMRStarEntry`](api/entry.md) is your gateway to a BMRB deposition. It wraps a
downloaded or local NMR-STAR file (`.str`) and exposes everything inside it as
tidy tables — no parsing or format knowledge required.

```python
entry = ms.NMRStarEntry.from_bmrb(25013)
```

From here you can pull **sequences** (what proteins/entities the entry
describes), **samples** (what conditions they were measured under), **spectrometers**
(the hardware), **citations**, **cross-references** (to PDB / AlphaFold), and
any measured **data** — chemical shifts, relaxation, order parameters, spectral
densities. See [Reading BMRB entries](guide/entries.md) for the full menu.

If `NMRStarEntry` doesn't have a dedicated method for something you need, two
escape hatches let you reach any field:

- `entry.categories()` — browse what saveframe categories the entry contains
- `entry.data_loop(category, loop_name, tags=None)` — extract any tabular loop as a DataFrame

## ChemicalShifts

[`ChemicalShifts`](api/chemshift.md) is a tidy table of assigned backbone shifts
— one row per atom. You build it from an `NMRStarEntry` or fetch it directly
from the BMRB:

```python
cs = ms.ChemicalShifts.from_bmrb(5363)
# or
entry = ms.NMRStarEntry.from_bmrb(5363)
cs = ms.ChemicalShifts.from_entry(entry)
```

`cs.data` is a DataFrame with columns `Seq_ID`, `Comp_ID`, `Atom_ID`,
`Atom_type`, and `Val` (the shift in ppm). It's the standard input for most
downstream analyses.

Two key operations live here:

- **Re-referencing** — correct mis-referenced shifts with `cs.reref("lacs")` or
  `"panav"`. See [Re-referencing](guide/rereferencing.md).
- **Building peak lists** — go straight from shifts to assigned peaks with
  `cs.peaklist()`.

## PeakList

[`PeakList`](api/peaklist.md) is an assigned peak table — usually an amide HSQC
(¹H–¹⁵N correlations), but any dimension pair you want. Build it from shifts
or read it from a CSV:

```python
peaks = cs.peaklist()                  # from ChemicalShifts
peaks = ms.PeakList.from_bmrb(5363)    # or direct from BMRB
peaks = ms.PeakList.from_csv("peaks.csv")
```

`peaks.data` is a DataFrame with one row per peak, columns for per-dimension ppm
values (e.g. `H_ppm`, `N_ppm`), and assignment labels. From here you can:

- **Compare peaks** — use `makeshift.spectra.map_peaklists` to align experimental
  and reference peaks, for instance in a titration or CSP analysis. See
  [Spectra](guide/spectra.md#aligning-peak-lists).
- **Plot assignments** — the `makeshift.spectra` plotting helpers take a PeakList
  and draw it on a spectrum.
- **Summarise completeness** — `peaks.assignment_string()` renders a compact
  per-residue label string (`'A'` assigned, `'.'` missing, `'P'` proline).

## The workflow

```python
import makeshift as ms

# 1. Get an entry and explore it
entry = ms.NMRStarEntry.from_bmrb(25013)
entry.datasets()       # what's in it
entry.sequences()      # the proteins

# 2. Extract chemical shifts, optionally re-reference
cs = ms.ChemicalShifts.from_entry(entry, reref="lacs")

# 3. Build a peak list
peaks = cs.peaklist()

# 4. Go further (relax dynamics, CPMG, structure prediction, etc.)
from makeshift.relaxation import RelaxationProfile
prof = RelaxationProfile.from_entry(entry)
prof.add_rigid_prediction()
prof.plot("R2_R1")
```

Each object is independent — you can use just `NMRStarEntry` for metadata, or
skip straight to `ChemicalShifts.from_bmrb` if shifts are all you need.
