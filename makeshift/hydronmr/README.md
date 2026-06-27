# hydronmr

Python port of Fast-HYDRONMR: predicts per-residue NMR relaxation (T1, T2,
T1/T2, NOE) from a protein PDB structure. Lives in the `makeshift` package as
`makeshift.hydronmr`.

## Quick start

```python
from makeshift.hydronmr import run

result = run("in.pdb")
```

`result.per_residue` is a dict keyed by `(chain, resseq)` with values
`(T1, T2, T1_over_T2, NOE)`.

`run()` takes two optional arguments:

- `config_path=` — selects a different `config.yml` (defaults to the bundled
  one next to `engine.py`).
- `csv_path=` — also writes a CSV (`resseq, T1, T2, T1_over_T2, NOE`).

## Getting the results as a DataFrame

The `Result` object exposes its own tidy views — no intermediate file needed:

```python
result = run("in.pdb")

df = result.to_dataframe()
#   chain  seqpos    T1    T2  T1_over_T2   NOE
#   one row per residue

result.to_csv("t1t2.csv")   # resseq, T1, T2, T1_over_T2, NOE
```

`to_dataframe()` is the general, correctly-labeled table; derive whatever a
given analysis needs from it rather than reaching for legacy column shapes.

### Predicted R2 (for R2-based analyses)

A transverse relaxation rate is simply `1 / T2`. For multi-chain structures,
`per_residue` is keyed per chain and can carry duplicate `seqpos`, so collapse
by mean:

```python
df = result.to_dataframe()
r2 = (df.assign(R2_hydro=1.0 / df["T2"])
        .groupby("seqpos", as_index=False)["R2_hydro"].mean())
```

This is the quantity the relaxation classifier uses as a *shape* template
(least-squares-scaled onto the measured R2 over rigid residues), which is why
the systematic scale offset noted below doesn't affect it.

## Layout

- `hydronmr/__init__.py` — re-exports `run` (the only public entry point).
- `hydronmr/engine.py` — `HydronmrState`, `Result` (with `to_dataframe()` /
  `to_csv()`), and `run()`.
- `hydronmr/physics/` — ported physics routines (structure/PDB parsing,
  hydrodynamic tensors, NMR relaxation, config loading, CSV output).
- `hydronmr/config.yml` — default run parameters (temperature, viscosity, AER,
  field strength, etc.); only the PDB path varies per call.

## Known limitations

- The IND=3 shell model (used by the original ground-truth runs) is not yet
  ported; `config.yml` defaults to `ind: 1` on the raw AER bead model, which
  reproduces per-residue T1/T2 *shape* (r > 0.97 vs ground truth across
  validated proteins) but with a systematic scale offset.
- Large structures (>~5000 atoms) will OOM with the current dense O(N²)
  mobility-matrix approach.