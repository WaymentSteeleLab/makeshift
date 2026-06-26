# hydronmr (Python port)

Python port of Fast-HYDRONMR for predicting per-residue NMR relaxation
(T1, T2, T1/T2, NOE) from a protein PDB structure.

## Quick start

```python
from hydronmr.engine import run

result = run("in.pdb")
```

`result.per_residue` is a dict keyed by `(chain, resseq)` with values
`(T1, T2, T1_over_T2, NOE)`.

Optional `config_path=` selects a different `config.yml` (defaults to
the bundled `python_port/config.yml`); optional `csv_path=` writes a
CSV (`resseq,T1,T2,T1_over_T2,NOE`).

## Getting a `read_hydronmr_t12`-shaped DataFrame

The old `.t12`-file pipeline produced a DataFrame with columns
`seqpos`, `R2_hydro`, `col3` (= T1/T2 ratio), built by
`read_hydronmr_t12`. To get the equivalent DataFrame directly from the
new API (no intermediate file):

```python
import pandas as pd
from hydronmr.engine import run

def hydronmr_t12(pdb_path, n_residues=None, config_path=None):
    kwargs = {}
    if config_path is not None:
        kwargs["config_path"] = config_path
    result = run(pdb_path, **kwargs)

    rows = [
        dict(seqpos=resseq, R2_hydro=t1, col3=ratio)
        for (chain, resseq), (t1, t2, ratio, noe) in result.per_residue.items()
    ]
    df = pd.DataFrame(rows).sort_values("seqpos").reset_index(drop=True)

    if n_residues is not None:
        df = df[df["seqpos"] <= n_residues].reset_index(drop=True)
    return df
```

Notes:
- `R2_hydro` here is actually **T1** (matching `col3` of the original
  `.t12`, which is `R2_hydro`/T1 -- check against your downstream usage;
  if you only care about `col3` (T1/T2), that column is correct as-is).
- `col3` = T1/T2, the primary quantity used in validation against
  ground truth.
- `n_residues` works the same as before: pass `len(sequence)` to drop
  any extra residues from a multi-chain PDB (e.g. crystallographic
  dimers), since `result.per_residue` is keyed per-chain and may
  contain duplicate `seqpos` values across chains -- filter/group by
  `chain` first if you need to disambiguate those before applying
  `n_residues`.
- All values are computed in-memory; no `.t12`/`.dat`/`.out` files are
  produced unless you pass `csv_path=` to `run()`.

## Layout

- `hydronmr/engine.py` -- `HydronmrState`, `Result`, and `run()` (the
  only public entry point).
- `hydronmr/physics/` -- ported physics routines (structure/PDB
  parsing, hydrodynamic tensors, NMR relaxation, config loading, CSV
  output).
- `config.yml` -- default run parameters (temperature, viscosity, AER,
  field strength, etc.); only the PDB path varies per call.
- `tests/test_regression.py` -- regression test pinning results
  against `GROUND_TRUTH_DONT_OVERWRITE/CYPA`.

## Known limitations

- IND=3 shell model (used by the original ground-truth runs) is not
  yet ported; `config.yml` defaults to `ind: 1` on the raw AER bead
  model, which reproduces per-residue T1/T2 *shape* (r > 0.97 vs
  ground truth across validated proteins) but with a systematic scale
  offset.
- Large structures (>~5000 atoms, e.g. YJBJ) will OOM with the current
  dense O(N^2) mobility-matrix approach.
