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

**28 single-field entries pass.** Pooled across all of them:

| Metric | Value |
|---|---|
| Pearson r | 0.785 |
| RMSE | 0.099 |
| n | 2656 residues |

Excluding "ambiguous"-model residues (both a NOE deficit and an R2 excess
detected -- S2 falls back to the Model 1 point estimate rather than being
confidently attributed to Model 2 or 3): r=0.799, RMSE=0.101, n=2271. See
[validate_s2_no_ambiguous.png](../demos/s2_validation/validate_s2_no_ambiguous.png)
alongside the all-models plot below.

### Per-entry breakdown

Sorted by Pearson r, high to low.

| BMRB | Entry | Field (MHz) | n | r | RMSE |
|---|---|---|---|---|---|
| 4366 | Stromelysin+inhibitor (S1-S3, b) | 600 | 145 | 0.976 | 0.033 |
| 17046 | Trp repressor, L75F (holo) | 600 | 73 | 0.960 | 0.075 |
| 6577 | IF2 C1-subdomain | 600 | 67 | 0.957 | 0.105 |
| 4364 | Stromelysin+inhibitor (S1-S3P) | 600 | 138 | 0.955 | 0.046 |
| 4365 | Stromelysin+inhibitor (S1-S3, a) | 600 | 141 | 0.954 | 0.036 |
| 5154 | N-TIMP-1, free | 500 | 102 | 0.946 | 0.069 |
| 5991 | Ets-1 deltaN301 | 500 | 124 | 0.933 | 0.055 |
| 4390 | Eotaxin | 500 | 57 | 0.932 | 0.082 |
| 17069 | E73, SSV-RH | 600 | 51 | 0.922 | 0.067 |
| 18758 | E1 enzyme half-domain | 400 | 71 | 0.915 | 0.097 |
| 5549 | NPY analog [31Ala,32Pro] | 500 | 28 | 0.913 | 0.167 |
| 17041 | Trp repressor, WT (holo) | 600 | 65 | 0.902 | 0.081 |
| 50212 | M. tuberculosis pyrophosphatase | 700 | 136 | 0.901 | 0.059 |
| 5153 | N-TIMP-1 + MMP-3 | 600 | 84 | 0.901 | 0.083 |
| 5550 | NPY, micelle-bound | 500 | 27 | 0.897 | 0.191 |
| 15451 | GABPa OST domain | 500 | 77 | 0.892 | 0.065 |
| 17010 | Trp repressor, WT | 600 | 55 | 0.890 | 0.064 |
| 17012 | Trp repressor, L75F | 600 | 81 | 0.850 | 0.108 |
| 17047 | Trp repressor, A77V (holo) | 600 | 72 | 0.847 | 0.096 |
| 26513 | MptpA, apo (b) | 600 | 111 | 0.827 | 0.069 |
| 18389 | CtCBM11 (b) | 600 | 148 | 0.820 | 0.082 |
| 19388 | MptpA, apo (a) | 600 | 113 | 0.806 | 0.070 |
| 17306 | NRC ankyrin repeat protein | 600 | 85 | 0.738 | 0.135 |
| 18388 | CtCBM11 (a) | 600 | 156 | 0.711 | 0.095 |
| 17013 | Trp repressor, A77V | 600 | 72 | 0.696 | 0.113 |
| 6470 | Ubiquitin | 600 | 63 | 0.638 | 0.088 |
| 4689 | Human growth hormone | 500 | 140 | 0.629 | 0.203 |
| 27890 | BlaC + clavulanic acid | 850 | 174 | 0.547 | 0.158 |

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

**Unreliable calibration (1 entry).** 50001 (ARR_CleD + c-di-GMP) —
`fit_order_parameters()` raises `model_free.FittingError` rather than
attempting a fit: only 8 presumed-rigid calibration residues, with the
calibrated tensor rate pinned at the search boundary and R1/R2 implying
substantially different order parameters for that calibration set (gap
+0.42, over the 0.3 unreliable threshold — see
[Calibration](guide/order-parameters.md)). No single overall tumbling
tensor reconciles R1 and R2 here, so no S² is reported rather than
reporting numbers that shouldn't be trusted.

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

**5. Model 2's tau_e search had a real second, unphysical minimum.** As
tau_e approaches the protein's own overall tumbling time tau_m, the
"fast"-motion term converges to the same functional form as the S²
term, so the fit stops depending on S² at all -- opening a genuine
double-well in chi2(tau_e) where large tau_e lets S² collapse toward 0.
A fixed absolute cap on the search (the old `TAU_E_MAX_S=3ns` alone)
doesn't protect any protein whose own tau_m isn't safely larger than
that -- true for most single-domain proteins. Found on a RelaxDB entry
(Protein G B1 domain, tau_m~4.7ns after calibration): 12.3% of Model-2
fits across that dataset had collapsed to S²<0.5, some as low as 0.25 in
a famously rigid, extensively characterized domain core. Fixed by
capping the search at `tau_m / model_free.TAU_E_SEPARATION_FACTOR`
(10x) in addition to the fixed constant — see that constant's docstring
for the direct chi2(tau_e) evidence. This entry's own BMRB validation
set didn't contain the pathology as severely as the RelaxDB case that
surfaced it, but the fix still shifts some Model-2 residues' S²: pooled
fit-vs-deposited r moved from 0.797 to 0.785 (a few entries improved --
6577 0.948->0.957 -- while one dropped more -- 17306 0.855->0.738 --
net close to a wash, and not indicative of the fix being wrong, since
the deposited fit isn't ground truth either).

A fixed fraction of tau_m isn't fully trustworthy on its own, either:
checked on other RelaxDB entries, several residues' true chi2(tau_e)
minimum legitimately sits close to (or past) that cap without being the
tau_e~tau_m degeneracy -- e.g. one entry's N-terminal residues had a
genuine minimum around 1-1.7ns that the `tau_m/10` cap alone would have
clipped. `model_free._fit_model2` now re-checks any residue landing at
that cap against a much wider one (`tau_m/2`): a genuine interior
minimum found there is accepted; a residue still climbing toward the
wider boundary is genuinely unresolvable and falls back to Model 1's S2
with `tau_e_degenerate=True` rather than reporting either a clipped or a
degenerate tau_e. Across the full RelaxDB set this triggers rarely (13
of 3330 Model-2 residues, across 9 of 119 fit entries) -- most
boundary-adjacent cases turn out to be genuine, not degenerate.

**6. A biased calibration inflates Model 3 (Rex) selection, and one
family of entries had the wrong oligomeric structure entirely.**
15 of the 28 validation entries have >20% of residues assigned Model 3
-- implausibly widespread real chemical exchange (e.g. 6470/ubiquitin at
88%, a famously rigid, extensively studied protein). Root cause: the
Rex test (`fit_residue`'s r2_flag) compares observed R2 against
`S2_from_R1 * R2_rigid`, but `S2_from_R1` inherits the same R1-vs-R2
tension `assess_calibration`'s `gap` already measures -- and unlike the
0.3 "unreliable" threshold, even a *mild* gap (0.07-0.23, well under
that threshold) is enough to make most of a protein's residues look
falsely R2-excess. Checked directly: median(implied Rex / R2_rigid)
across every residue in an entry tracks that entry's calibration gap
almost exactly (e.g. 6470: gap=+0.225, ratio=+0.218) -- the signature of
a protein-wide bias, not per-residue exchange. Not yet fixed (candidate
fix: correct the Rex-test baseline by the measured gap before testing
for excess).

Separately, for BMRB 17012 (and 17010, 17013 -- all Trp repressor
mutants), the *structural* cause is more fundamental: the resolved PDB
(3WRP) is only the crystallographic asymmetric unit -- RCSB's own
`pdbx_struct_assembly.oligomeric_count=2` says the biological assembly
is dimeric, but the file has just one chain, with the second protomer
only implied by a BIOMT symmetry transform this code never applies.
Trp repressor is a constitutive homodimer; modeling it as a monomer
gives HYDRONMR the wrong diffusion tensor size *and shape*, which the
rate-only calibration (`shape_bounds=(1.0, 1.0)`, see
`model_free.calibrate_tensor`) can't correct by design.
`RelaxationProfile._check_biological_assembly` now warns whenever a
`source="rcsb"` structure is missing chains the assembly needs (a
single lightweight RCSB API call, checked against the resolved
structure's own chain count) -- confirmed to fire only for 3WRP among
all 28 entries, and to stay silent for every correctly-monomeric
structure checked (1D3Z/ubiquitin, 2L6B/ankyrin repeat, 2JUO/GABPa).

That still doesn't fully explain the pattern, though: 17046 and 17047
(the other two Trp-repressor mutants) resolve to a *correctly* complete
dimer (2XDI, both chains present) and still show 47-51% Model 3, with
unusually large tau_scale corrections (~2.0-2.1x, vs ~1-1.5x typical for
monomeric entries) that don't track their calibration gap as cleanly as
the monomeric cases did. That looks like a separate, harder problem --
the uniform-bead hydrodynamic approximation may simply be less accurate
for larger multi-chain assemblies than for compact single domains, not
something a data/structure-resolution fix can address. Left open.

## Reproducing this validation

`demos/s2_validation/validate_s2.py` fits every entry in its `ENTRIES`
list, compares against deposited backbone S² (`Atom_ID` N/H, filtering out
methyl/side-chain order parameters), and writes `validate_s2_results.csv`
(per-entry stats), `validate_s2_pairs.csv` (every fit/deposited S² pair),
and three figures — a pooled scatter + per-entry correlation bar chart
(`validate_s2.png`), the same pair excluding "ambiguous"-model residues
(`validate_s2_no_ambiguous.png`), and a small-multiples grid, one panel
per protein (`validate_s2_grid.png`). It needs network access (to fetch
entries and structures) and takes on the order of 15–20 minutes for the
full set.
