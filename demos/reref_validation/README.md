Outputs were generated on August 5, 2026.

PANAV was v 2.1, run on NMRbox
LACS is from LACS validation files hosted on BMRB

Note that not all entries have LACS validation (if they don't have carbons). All proteins are the same as those used for RCI validation.

## Validation script

`reref_validation.py` compares makeshift's public `ChemicalShifts.reref`
offsets against the bundled LACS and PANAV reference dumps for the 9 local
deposits in `inputs/`.

```bash
python demos/reref_validation/reref_validation.py
```

Produces `reref_validation.png` (reference vs. makeshift scatter for each
method). Offsets use the paper convention
(`offset ≈ d_ave − d_obs`, `corrected = Val + offset`), matching the LACS /
PANAV dumps directly (LACS r ≈ 0.996, PANAV r ≈ 0.96 on this set).
