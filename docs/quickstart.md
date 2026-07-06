# Quickstart

This page walks through the core workflow: fetch a BMRB entry, get tidy chemical
shifts, optionally re-reference them, and build an assigned peak list. The same
example lives in `demos/quick_start.ipynb`.

## Fetch and parse shifts

[`ChemicalShifts.from_bmrb`](api/chemshift.md) downloads a BMRB entry and returns
a tidy table — one row per observed shift.

```python
import makeshift as ms

cs = ms.ChemicalShifts.from_bmrb(5363)

cs.data          # one row per shift: Seq_ID, Comp_ID, Atom_ID, Atom_type, Val
cs.sequences()   # one row per entity: ID, polymer type, one-letter sequence
```

The columns are deliberately tidy so you can go straight to pandas:

| Column | Meaning |
|---|---|
| `Seq_ID` | Residue number in the entity sequence |
| `Comp_ID` | Residue type (three-letter, e.g. `ALA`) |
| `Atom_ID` | Atom name (e.g. `CA`, `HN`, `N`) |
| `Atom_type` | Element (`C`, `H`, `N`, …) |
| `Val` | Chemical shift in ppm |

## Re-reference shifts

BMRB shifts are sometimes mis-referenced — a constant offset shifts every peak of
a given nucleus. Pass `reref=` to correct this on load, and optionally compute the
chemical-shift index (CSI):

```python
cs = ms.ChemicalShifts.from_bmrb(4527, reref="lacs", calc_csi=True)
cs.reref_offsets   # {atom: offset applied}
```

See [Re-referencing](guide/rereferencing.md) for the `"lacs"` vs `"panav"`
methods.

## Build a peak list

From an assigned shift table you can synthesize an assigned peak list — by
default an amide HSQC (¹H–¹⁵N):

```python
peaks = cs.peaklist()
peaks.data
```

You can also build a peak list directly from an entry or BMRB id, or read one
from a CSV — see [Peak lists](guide/peaklists.md).

## Go deeper

- Explore an entry's full contents (samples, spectrometers, relaxation, citation)
  with [`NMRStarEntry`](guide/entries.md).
- Turn deposited relaxation into a per-residue dynamics profile with
  [`RelaxationProfile`](guide/relaxation.md).
- Run a full [CPMG dispersion pipeline](guide/cpmg.md) from `.ucsf` planes.
- Predict dynamics and structure from shifts with
  [TALOS-N](guide/talosn.md) or from a PDB with [HYDRONMR](guide/hydronmr.md).
