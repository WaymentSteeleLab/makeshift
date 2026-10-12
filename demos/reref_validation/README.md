# Re-referencing validation

Live comparison of makeshift LACS / PANAV offsets against BMRB-hosted
references. Progress is **resumable**: each finished entry is appended to
`reref_validation_results.jsonl`, so a killed/rerun job skips those ids.

| Method | BMRB source |
|--------|-------------|
| LACS   | `…/bmr{id}/validation/LACS.str` |
| PANAV  | `https://api.bmrb.io/current/entry/{id}/validate` |
| Shifts | `ChemicalShifts.from_bmrb(..., keep_download=False)` |

```bash
# corpus run (resumes automatically if results JSONL exists)
python demos/reref_validation/reref_validation.py --all --prefilter-lacs --workers 48

# same command again → skips finished ids, continues the rest
python demos/reref_validation/reref_validation.py --all --workers 48

# LACS only (much faster)
python demos/reref_validation/reref_validation.py --all --methods lacs --workers 16

# ignore prior results (use a new --results path for a clean file)
python demos/reref_validation/reref_validation.py --all --no-resume --results /tmp/fresh.jsonl
```

**Recomputing makeshift offsets without re-fetching PANAV.** The validate
API is the slow step. After changing makeshift (e.g. `panav_distns.csv`), reuse
the BMRB PANAV references stored in a prior results JSONL and write to a new
results file (otherwise resume skips every id):

```bash
python demos/reref_validation/reref_validation.py --all \
    --panav-refs-from demos/reref_validation/reref_validation_results.jsonl \
    --results demos/reref_validation/reref_validation_results_v2.jsonl
```

Ids in that file use their stored PANAV refs; ids not in it are fetched as
usual. LACS and shifts are still fetched. Caveats: only atoms that were paired
in the old run have stored refs, and entries with an empty `panav` list
(no HA, no PANAV data, or a past fetch failure) are not retried.

**On-disk artifacts** (beside this script by default):

| File | Role |
|------|------|
| `bmrb_all_ids.txt` | Cached entry list for `--all` |
| `reref_validation_results.jsonl` | Per-entry offsets (resume source of truth) |
| `reref_validation.png` | Paper figure: LACS (3, centered) over PANAV (4) |
| `reref_validation_rows.tsv` | Flat table for plotting / Excel |

`--refresh-ids` rebuilds the id list. `--no-resume` does not skip finished ids.
`--panav-refs-from` reuses PANAV references from a prior results JSONL.
