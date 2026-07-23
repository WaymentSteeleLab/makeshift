# hydronmr validation

Validates `makeshift.hydronmr.run()` (see [`makeshift/hydronmr/README.md`](../../makeshift/hydronmr/README.md))
against real output from the original **Fast-HYDRONMR** Fortran program, on 7
small/medium proteins.

## What's here

- **`GROUND_TRUTH_DONT_OVERWRITE/`** — 8 protein directories (`AQADK`, `BLAC`,
  `BLVRB`, `CHI19`, `CYPA`, `KRAS`, `VHR`, `YJBJ`), each containing the actual
  input/output of a real Fast-HYDRONMR run:
  - `in.pdb` — the input structure.
  - `hydronmr.dat` — the run parameters the Fortran program used (AER bead
    radius, temperature, viscosity, field strength, etc). These are *not* all
    identical across proteins — notably **BLAC** was run with AER=2.0 Å and
    B0=14.1 T instead of the usual 3.0 Å / 11.74 T — so the validation script
    reads this file per-protein rather than assuming one fixed config.
  - `tmp.t12` — the original per-residue T1/T2 output (3 header lines, then
    `resseq, T1/T2, <unused column>` rows; blank rows are residues the
    original run excluded — N-terminus, proline, missing density).
  - `tmp.res`, `output`, `tmp-pri.bea`, `tmp-pri.vrml` — other original-run
    artifacts (full text report, bead model, VRML visualization), not used by
    the validation script but kept for reference.
  - **Do not overwrite/regenerate these** — they're the one fixed reference
    point external to this codebase (an independent binary's output), which
    is the entire point of keeping them checked in.
  - `YJBJ` (23001 atoms) is present but excluded from validation: the current
    `makeshift.hydronmr` mobility-matrix implementation is dense O(N²) and
    runs out of memory on a structure this large.

- **`validate_t1t2_meansub.py`** — runs `makeshift.hydronmr.run()` on each
  protein's `in.pdb` (with that protein's own `hydronmr.dat` parameters),
  parses the matching `tmp.t12`, and compares per-residue T1/T2.

- **`validate_t1t2_meansub.png`** — the output figure: one row per protein,
  each showing "T1/T2 − mean(T1/T2)" vs. residue index for both the original
  Fortran output and the makeshift prediction, titled with `N` (number of
  residues both runs produced a value for) and Pearson `r` between the two
  series.

## Why mean-subtracted, not raw T1/T2

`makeshift.hydronmr` currently approximates the hydrodynamic bead model as
one bead per heavy atom, all with a uniform radius (the "AER" mode) — it does
not yet implement the shell model the ground-truth runs actually used
(`IND=3`). That gives a systematic scale offset between predicted and
original absolute T1/T2 values, but reproduces the *shape* — which residues
are more/less mobile relative to the rest of the protein — well. Subtracting
each series' own mean per protein cancels that offset and isolates the
shape comparison, which is what the Pearson `r` in each panel title reflects
(r = 0.92–0.9996 across the 7 proteins). See "Known limitations" in
[`makeshift/hydronmr/README.md`](../../makeshift/hydronmr/README.md) for more
on the scale-offset caveat and where it matters (or doesn't) downstream.

## Running it

```bash
python demos/hydronmr_validation/validate_t1t2_meansub.py
```

Prints `N` and `r` per protein to stdout and (re)writes
`validate_t1t2_meansub.png` in this directory.
