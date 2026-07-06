# Chemical shifts

[`ChemicalShifts`](../api/chemshift.md) is a tidy table of assigned chemical
shifts — one row per atom — plus the operations you most often want on them:
re-referencing, the chemical-shift index (CSI), and building peak lists.

## Constructing

```python
import makeshift as ms

# From a BMRB id (downloads + parses)
cs = ms.ChemicalShifts.from_bmrb(5363)

# From an already-parsed entry
entry = ms.NMRStarEntry.from_bmrb(5363)
cs = ms.ChemicalShifts.from_entry(entry)
```

Both constructors accept `reref=` and `calc_csi=` so you can correct referencing
and compute CSI in one step:

```python
cs = ms.ChemicalShifts.from_bmrb(4527, reref="lacs", calc_csi=True)
```

## The table

```python
cs.data
```

| Column | Meaning |
|---|---|
| `Seq_ID` | Residue number |
| `Comp_ID` | Residue type (three-letter) |
| `Atom_ID` | Atom name |
| `Atom_type` | Element |
| `Val` | Shift (ppm) |

After `calc_csi=True` (or `cs.add_csi()`), two more columns appear: `csi_raw`
(secondary shift) and `csi` (the discretised ±1/0 index).

## Re-referencing

```python
cs.reref(method="panav")   # or "lacs"
cs.reref_offsets           # {'N': ..., 'CA': ..., 'CB': ..., ...}
```

The two methods and when to prefer each are covered in
[Re-referencing](rereferencing.md).

## Chemical-shift index

```python
cs.add_csi()   # adds csi_raw and csi columns in place; returns self
```

CSI compares each backbone shift to random-coil values to flag helix/strand
propensity per residue — the basis PANAV uses to pick a reference distribution.

## From shifts to peaks

```python
peaks = cs.peaklist()                       # default amide HSQC
peaks = cs.peaklist(dims=[("H", "N")])      # explicit dimensions
peaks = cs.peaklist(entity_id=1)            # a specific entity
```

See [Peak lists](peaklists.md) for the full set of dimension options and outputs.

## Sequences

```python
cs.sequences()          # one row per entity
cs.get_entry()          # the underlying NMRStarEntry, if built from one
```

## Full API

See the [`ChemicalShifts` reference](../api/chemshift.md).
