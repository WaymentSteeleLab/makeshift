# HYDRONMR prediction

`makeshift.hydronmr` is a pure-Python port of Fast-HYDRONMR: it predicts
per-residue NMR relaxation (T1, T2, T1/T2, NOE) from a protein PDB structure by
computing a rigid-body diffusion tensor. It's the engine behind the rigid-body
comparison in a [relaxation profile](relaxation.md).

## Quick start

```python
from makeshift.hydronmr import run

result = run("in.pdb")
```

`run()` takes two optional arguments:

- `config_path=` — a different `config.yml` (defaults to the one bundled next to
  the engine).
- `csv_path=` — also write a CSV (`resseq, T1, T2, T1_over_T2, NOE`).

## Getting results

The [`Result`](../api/hydronmr.md) object exposes tidy views — no intermediate
file needed:

```python
result = run("in.pdb")

df = result.to_dataframe()
#   chain  seqpos    T1    T2  T1_over_T2   NOE
#   one row per residue

result.to_csv("t1t2.csv")     # resseq, T1, T2, T1_over_T2, NOE
```

The raw mapping is also available:

```python
result.per_residue          # {(chain, resseq): (T1, T2, T1_over_T2, NOE)}
```

`to_dataframe()` is the general, correctly-labelled table; derive whatever a
given analysis needs from it.

## Where the structure comes from

`run()` takes a local PDB path. To fetch a structure by PDB id or UniProt
accession first, use [`fetch_structure`](datasets.md#structures):

```python
from makeshift.utils import fetch_structure

pdb = fetch_structure("1WRP")        # RCSB
pdb = fetch_structure("P0DP23")      # AlphaFold DB
result = run(pdb)
```

## Full API

See the [HYDRONMR reference](../api/hydronmr.md).
