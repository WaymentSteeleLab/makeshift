# Re-referencing

BMRB shifts are sometimes mis-referenced — a constant offset shifts every peak of
a given nucleus. [`ChemicalShifts.reref`](../api/chemshift.md) corrects this in
place using one of two published methods.

## Usage

```python
import makeshift as ms

cs = ms.ChemicalShifts.from_bmrb(4527)
cs.reref(method="panav")     # or "lacs"
print(cs.reref_offsets)      # {'N': ..., 'CA': ..., 'CB': ..., ...}
```

Or apply it on load:

```python
cs = ms.ChemicalShifts.from_bmrb(4527, reref="lacs")
```

`reref_offsets` holds the offset applied to each nucleus, so the correction is
fully transparent and reversible.

## The two methods

### PANAV 

**PANAV** ([Wang & Wishart 2005](https://pubmed.ncbi.nlm.nih.gov/15772753/))
    uses rarely-misreferenced HA shifts to assign secondary structure, then aligns
    N/CA/CB to curated per-structure reference distributions
    ([Wang & Jardetzky 2002](https://onlinelibrary.wiley.com/doi/10.1110/ps.3180102)).

```python
cs.reref(method="panav")
```

### LACS
**LACS** ([Wang & Markley 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2782637/))
    fits secondary shift vs. CSI so the random-coil regime intercepts at the
    origin; it covers CA, CB, C′, N, and HN.

```python
cs.reref(method="lacs")
```

For example, entries 6586 and 4150 have been described in the literature as needing re-referencing.

![Re-referencing example](../static/example_rereferencing_ed.png)

## Under the hood

The [`makeshift.reref`](../api/reref.md) subpackage exposes the underlying
routines if you want to work with raw DataFrames rather than a `ChemicalShifts`
object, you can use `compute_offsets`, `apply_offsets`, `reref_lacs`, and `reref_panav`.

## Full API

See the [re-referencing reference](../api/reref.md).
