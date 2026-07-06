# Peak lists

A [`PeakList`](../api/peaklist.md) is an assigned peak table — most often an
amide HSQC (¹H–¹⁵N) synthesized from assigned chemical shifts, but it can also be
read from a CSV or built for arbitrary dimension pairs.

## Building a peak list

### "From chemical shifts"

```python
import makeshift as ms

cs = ms.ChemicalShifts.from_bmrb(5363)
peaks = ms.PeakList.from_chemshifts(cs)     # or simply cs.peaklist()
```

### "From an entry / BMRB id"

```python
peaks = ms.PeakList.from_bmrb(5363)

# or

entry = ms.NMRStarEntry.from_bmrb(5363)
peaks = ms.PeakList.from_entry(entry)
```

=== "From a CSV"

    ```python
    peaks = ms.PeakList.from_csv("peaks.csv", seq_offset=0)
    ```

## Choosing dimensions

By default a peak list is the amide HSQC. Pass `dims=` to build other
correlations. Each dimension pair names the two atoms correlated in a peak:

```python
peaks = cs.peaklist(dims=[("H", "N")])   # amide HSQC (default)
```

Select a single entity in a multi-entity complex with `entity_id=`, or a
specific chemical-shift saveframe with `cs_saveframe=`.

## The table

```python
peaks.data
```

Columns include the per-dimension ppm values (e.g. `H_ppm`, `N_ppm`) and an
assignment label, one row per peak.

## Assignment strings

`assignment_string` renders a compact per-residue label string against a
sequence — `'A'` for an assigned residue, `'.'` for a missing one, and `'P'` for
proline:

```python
peaks.assignment_string()
peaks.assignment_string(entity_id=1)
```

This is handy for a quick, at-a-glance completeness check of an assignment.

## Aligning two peak lists

To match an experimental peak list against a reference assignment (for example
before a chemical-shift-perturbation analysis), use
[`map_peaklists`](../api/spectra.md) from `makeshift.spectra`. See
[Spectra](spectra.md#aligning-peak-lists).

## Full API

See the [`PeakList` reference](../api/peaklist.md).
