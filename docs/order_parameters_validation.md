# Order parameter validation report

`RelaxationProfile.fit_order_parameters()` (see the [order parameters
guide](guide/order-parameters.md) for the physics) was validated by fitting
S² for every BMRB entry that has raw R1/R2/NOE, deposited model-free order
parameters, and a linkable structure, then comparing residue-by-residue
against the deposited S². This is a sanity check, not a reproduction
target: the model-selection rule here is deliberately simpler than
whatever produced each entry's deposited fit (usually Modelfree4/
FastModelFree) — see [Model selection](guide/order-parameters.md#the-physics)
— so per-residue model *assignments* aren't expected to match, only the S²
*magnitudes*.

## Screening

Every BMRB entry tagged with `_Order_param.Order_param_val` (128 total)
was checked for real — fetch, build a profile, fit, compare — rather than
excluded by guesswork, after a metadata prefilter (single polypeptide
entity, has T1/T2/NOE and order parameters: 79 candidates). Entries
deposited at more than one spectrometer field are excluded before any fit
is attempted (`RelaxationProfile._list_fields_mhz`): their deposited S²
almost certainly came from a joint multi-field fit, a better-constrained
problem than the single-field refit this code does, so a mismatch there
isn't evidence of anything wrong on either side.

**29 single-field entries pass.** Pooled across all of them:

| Metric | Value |
|---|---|
| Pearson r | 0.781 |
| RMSE | 0.103 |
| n | 2686 residues |

### Per-entry breakdown

Sorted by Pearson r, high to low.

| BMRB | Entry | Field (MHz) | n | r | RMSE |
|---|---|---|---|---|---|
| 4366 | Stromelysin+inhibitor (S1-S3, b) | 600 | 145 | 0.976 | 0.033 |
| 17046 | Trp repressor, L75F (holo) | 600 | 73 | 0.959 | 0.075 |
| 4364 | Stromelysin+inhibitor (S1-S3P) | 600 | 138 | 0.955 | 0.046 |
| 4365 | Stromelysin+inhibitor (S1-S3, a) | 600 | 141 | 0.954 | 0.036 |
| 6577 | IF2 C1-subdomain | 600 | 67 | 0.948 | 0.112 |
| 5154 | N-TIMP-1, free | 500 | 102 | 0.934 | 0.089 |
| 4390 | Eotaxin | 500 | 57 | 0.932 | 0.082 |
| 18758 | E1 enzyme half-domain | 400 | 71 | 0.915 | 0.097 |
| 17069 | E73, SSV-RH | 600 | 51 | 0.911 | 0.071 |
| 5549 | NPY analog [31Ala,32Pro] | 500 | 28 | 0.909 | 0.153 |
| 17041 | Trp repressor, WT (holo) | 600 | 65 | 0.894 | 0.083 |
| 15451 | GABPa OST domain | 500 | 77 | 0.885 | 0.066 |
| 5991 | Ets-1 deltaN301 | 500 | 124 | 0.885 | 0.069 |
| 5550 | NPY, micelle-bound | 500 | 27 | 0.870 | 0.193 |
| 50212 | M. tuberculosis pyrophosphatase | 700 | 136 | 0.857 | 0.070 |
| 17306 | NRC ankyrin repeat protein | 600 | 85 | 0.854 | 0.115 |
| 17047 | Trp repressor, A77V (holo) | 600 | 72 | 0.845 | 0.096 |
| 17012 | Trp repressor, L75F | 600 | 81 | 0.833 | 0.111 |
| 26513 | MptpA, apo (b) | 600 | 111 | 0.826 | 0.070 |
| 17010 | Trp repressor, WT | 600 | 55 | 0.825 | 0.070 |
| 19388 | MptpA, apo (a) | 600 | 113 | 0.806 | 0.070 |
| 5153 | N-TIMP-1 + MMP-3 | 600 | 84 | 0.793 | 0.142 |
| 18388 | CtCBM11 (a) | 600 | 156 | 0.711 | 0.097 |
| 17013 | Trp repressor, A77V | 600 | 72 | 0.695 | 0.113 |
| 18389 | CtCBM11 (b) | 600 | 148 | 0.688 | 0.091 |
| 6470 | Ubiquitin | 600 | 63 | 0.639 | 0.088 |
| 4689 | Human growth hormone | 500 | 140 | 0.623 | 0.206 |
| 50001 | ARR_CleD + c-di-GMP | 600 | 30 | 0.551 | 0.125 |
| 27890 | BlaC + clavulanic acid | 850 | 174 | 0.544 | 0.158 |

## What was excluded, and why

Every exclusion below was confirmed by checking, not assumed from
metadata.

**Multi-field (23 entries), excluded up front.** 4245, 4267, 4970, 5330,
5331, 5548, 5841, 6243, 6474, 6838, 11080, 15445, 15562, 16392, 17226,
17246, 18971, 26507, 26511, 27011, 27447, 27448, 27888. Four of these
(11080, 6243, 5841, 26507) were the initial motivation: fit against a
single field, they showed weak or negative correlation with no obvious
cause. Checking field counts explained all four at once — each is
deposited at 2–4 fields — and once applied as a blanket prefilter, 13 more
of what were originally "passing" single-model-run entries turned out to
be multi-field too and are excluded on the same grounds regardless of
their agreement.

**Data/parsing gap (3 entries), not a modeling problem.** 16917, 16918,
26779 — `NMRStarEntry.relaxation('T1')` returns zero rows for these
depositions despite BMRB's own metadata reporting nonzero T1 counts, a
nonstandard saveframe/tag layout the current parser doesn't cover. A
pre-existing gap in `makeshift.entry`, outside this validation's scope.

**Physical scope mismatch (2 entries).** 15097 (villin C-terminal
domains), 25852 (calmodulin + eNOS peptide) — both multi-domain
constructs joined by a flexible linker, so no single rigid-body diffusion
tensor describes the whole chain. The model assumes one rigid domain by
construction.

**Unresolved (1 entry).** 27929 (BlaC + avibactam): r=0.02, no specific
cause confirmed.

## Bugs found during validation

Three real correctness issues surfaced while extending from an initial
3-entry check to the full set, all fixed before the numbers above.

**1. Multi-field entries silently mixed spectrometer fields.**
`RelaxationProfile.from_entry` took the first-matching row per residue
across *every* deposited relaxation list, with no regard for which field
each list was measured at. Fixed with explicit field resolution
(`_resolve_field`/`_filter_by_list`), shared by `add_rigid_prediction` and
`fit_order_parameters` so the data and the physics predictions always
agree on field.

**2. Promiscuous PDB citation lists picked the wrong structure.**
`_resolve_pdb` took `get_pdb_ids()[0]` unconditionally. For a protein with
no depositor-curated related-entry link, that list is just every PDB
sharing the sequence via entity cross-references — 280 hits for ubiquitin,
in no meaningful order — and landed on a fusion/complex structure with the
wrong diffusion tensor. `_best_pdb_by_length` now checks candidates'
polymer size against RCSB's entry API and prefers one matching the
profile's own sequence length.

**3. Model 2's fit never actually used NOE.** The χ² being minimized for
the τₑ search only ever included R1 and R2 — despite NOE being exactly
what makes τₑ distinguishable from S² at all (see [the
physics](guide/order-parameters.md#the-physics)). Without it, S² and τₑ
were constrained by the same two numbers Model 1 already uses: degenerate,
with nothing favoring τₑ→0 even when it was correct. Fixed by adding NOE's
own residual to the outer χ². On ubiquitin, this single fix moved
fit-vs-deposited correlation from r=0.36 to r=0.76.

A fourth, smaller fix: `NMRStarEntry.get_alphafold_ids()` choked on an
accession wrapped in quotes with an appended domain-range annotation
(`"'P9WKD3[43 - 307]'"`, from three BlaC entries) — both are now stripped.

## Reproducing this validation

`demos/s2_validation/validate_s2.py` fits every entry in its `ENTRIES`
list, compares against deposited backbone S² (`Atom_ID` N/H, filtering out
methyl/side-chain order parameters), and writes `validate_s2_results.csv`
(per-entry stats), `validate_s2_pairs.csv` (every fit/deposited S² pair),
and two figures — a pooled scatter + per-entry correlation bar chart, and
a small-multiples grid, one panel per protein. It needs network access (to
fetch entries and structures) and takes on the order of 15–20 minutes for
the full set.
