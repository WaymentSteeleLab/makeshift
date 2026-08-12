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

After `calc_csi=True` (or `cs.add_csi()`), Wishart CSI columns appear —
see [Chemical-shift index](#chemical-shift-index) below.

## Re-referencing

```python
cs.reref(method="panav")   # or "lacs"
cs.reref_offsets           # {'N': ..., 'CA': ..., 'CB': ..., ...}
```

The two methods and when to prefer each are covered in
[Re-referencing](rereferencing.md).

## Chemical-shift index

```python
cs.add_csi()                         # default: wishart_94 (consensus)
cs.add_csi(method="wishart_92")      # ¹Hα only (Wishart et al. 1992)
cs.add_csi(method="wishart_94_ca")   # 13Ca only (Wishart & Sykes 1994)
cs.add_csi(method="wishart_94_cb")   # 13Cb (strand-only)
cs.add_csi(method="wishart_94_c")    # 13C'
```

Or in one shot: `ChemicalShifts.from_bmrb(4527, calc_csi=True)` /
`calc_csi="wishart_92"`.

| `method` | What it does |
|---|---|
| `"wishart_94"` (default) | HA+CA+CB+C′ indices (`csi_ha` …), density-filtered SS, majority consensus `ss` / `csi` |
| `"wishart_92"` | 1Ha CSI (1992) |
| `"wishart_94_ha"` / `"_ca"` / `"_cb"` / `"_c"` | Single-nucleus 1994 protocol |

Ranges are in `csi_wishart.csv`. Per-residue detail is on `cs.csi_table`.
makeshift does **not** implement CSI 2.0 / 3.0. LACS re-referencing uses its
own continuous CA−CB secondary shift, separate from this API.

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
