# Re-referencing validation

Live comparison of makeshift LACS / PANAV offsets against BMRB-hosted
references. Nothing is cached on disk except the output figure.

| Method | BMRB source |
|--------|-------------|
| LACS   | `…/bmr{id}/validation/LACS.str` |
| PANAV  | `https://api.bmrb.io/current/entry/{id}/validate` |
| Shifts | `ChemicalShifts.from_bmrb(..., keep_download=False)` |

```bash
# quick default set (9 RCI proteins)
python demos/reref_validation/reref_validation.py

# every macromolecule entry that has a LACS.str (~hours, LACS only — fast path)
python demos/reref_validation/reref_validation.py --all --methods lacs --workers 16

# every macromolecule entry, LACS + PANAV (PANAV validate is slow; many hours)
python demos/reref_validation/reref_validation.py --all --prefilter-lacs --workers 12

# custom id list
python demos/reref_validation/reref_validation.py --ids-file my_ids.txt --workers 8
```

`--all` pulls ~18k ids from `api.bmrb.io/v2/list_entries?database=macromolecules`.
`--prefilter-lacs` HEAD-scans for `LACS.str` first (recommended for corpus runs).
`--methods lacs` skips the slow PANAV validate API.

Writes `reref_validation.png`: LACS row (CA, CB, CO) and PANAV row
(N, CA, CB, CO), Makeshift on x, BMRB on y.

N is compared for PANAV only — BMRB `LACS.str` does not deposit N/HN.
