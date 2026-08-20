# Model-free order parameters

`RelaxationProfile.fit_order_parameters()` fits a per-residue generalized
order parameter S² — and, where the data support it, an internal
correlation time τₑ or an exchange term Rₑₓ — from R1/R2/NOE, the way
classic model-free software (Modelfree4/FastModelFree) does. The difference
is scope and reuse: it targets exactly the regime most deposited datasets
are actually in (a single field, one rigid domain), and it gets its
anisotropic-tumbling physics for free from the same rigid-body diffusion
tensor [`add_rigid_prediction()`](relaxation.md#the-rigid-body-prediction)
already computes via [HYDRONMR](hydronmr.md).

## How does this compare to running fast-ModelFree?

Full model-free analysis fits a nested hierarchy of five
models (S²; S²+τₑ; S²+Rₑₓ; S²+τₑ+Rₑₓ; S²f+S²s+τₑ) and picks the simplest
one an F-test can't reject. Models 4 and 5 have three free parameters —
against R1/R2/NOE at a single field, that's exactly- or over-parameterized,
and in practice the extra parameters trade off against each other in
nearly-degenerate directions (τₑ vs. Rₑₓ, or S²f vs. S²s) rather than
converging on a physical answer. Most deposited BMRB relaxation datasets
*are* single-field. So this fits only Models 1–3, and decides between them
from which residual each extra parameter physically explains, rather than
a percentile/F-test cascade built for arbitrating among five candidates.

The other half is the anisotropic tumbling itself. Model-free formalism
needs a diffusion tensor to build the spectral density from; makeshift
already computes one — the same rigid hydrodynamic bead-model tensor and
per-residue Woessner mode decomposition (five correlation times, five
orientation-dependent amplitudes) that
[`add_rigid_prediction()`](relaxation.md#the-rigid-body-prediction) uses
for its rigid-body R2/R1 prediction. `fit_order_parameters()` layers
internal motion on top of that same per-residue spectral density instead
of assuming isotropic tumbling or rebuilding the tensor machinery.

## The physics

Lipari & Szabo (1982) split a bond vector's reorientation into overall
tumbling and internal motion, assumed independent and separated in
timescale, so the total correlation function factorizes:
C(t) = C₀(t)·C_I(t), with C_I(t) = S² + (1−S²)e^(−t/τₑ). Under anisotropic
tumbling, C₀(t) isn't single-exponential — diagonalizing the rotational
diffusion tensor gives five correlation times τₖ (Woessner, 1962), each
weighted by an amplitude Aₖ set by the bond's orientation relative to the
tensor's principal axes. Combining the two gives the general model-free
spectral density:

```
J(ω) = Σₖ Aₖ · (2/5) · [ S²·τₖ/(1+ω²τₖ²) + (1−S²)·τₖ′/(1+ω²τₖ′²) ]
1/τₖ′ = 1/τₖ + 1/τₑ
```

which is exactly what
[`makeshift.relaxation.model_free.spectral_density`](../api/relaxation.md)
implements, and which reduces to the rigid-body J(ω) already used for
`add_rigid_prediction()` when S²=1, τₑ=0.

**Models 1 and 3 (τₑ→0).** In this limit both terms of J(ω) collapse onto
the same τₖ, so J(ω) = S²·J_rigid(ω) — the existing rigid prediction scaled
by one number. R1 and R2 become exactly linear in S², so it's solved by
weighted linear least squares against R1/R2, no nonlinear optimization
needed. A direct consequence: NOE, which enters only as a ratio with R1,
has S² cancel out of it completely in this limit. **NOE carries no
information about the size of S² here** — its only role is diagnosing
whether the τₑ→0 assumption itself holds: a NOE depressed below the
parameter-free rigid-tumbling prediction means it doesn't. Model 3 adds
exchange broadening as a term on R2 alone (R2 = S²·R2,rigid + Rₑₓ), with S²
fixed from R1 (which Rₑₓ doesn't touch) and Rₑₓ read off the R2 residual.

**Model 2 (finite τₑ).** Once τₑ is nonzero, the S² and (1−S²) terms have
different frequency dependence and no longer cancel in NOE — it becomes a
real third constraint, which is what makes τₑ distinguishable from S² at
all (two unknowns, three observables, one residual field is enough). τₑ is
found by a 1D search (the only place it enters nonlinearly); at each trial
τₑ, S² and the NOE residual are evaluated together, since fitting only
R1/R2 here would just be re-deriving Model 1's degeneracy.

**Model selection** follows from which residual each extra parameter
explains, not a formal statistical cascade: a NOE deficit relative to the
rigid prediction (parameter-free, no fit involved) implicates τₑ → Model 2;
an R2 excess beyond what the R1-derived S² already explains implicates
exchange → Model 3; a residue showing *both* signals is outside this
two-parameter scope and reported as `"ambiguous"` rather than forced into
either model.

**Calibration.** The bead model's diffusion-tensor *anisotropy shape* is
validated separately (`demos/hydronmr_validation`); its *absolute*
timescale is a known approximation (uniform-radius beads, not the real
AtoB table). Left uncorrected this pushes fitted S² outside [0, 1] for
obviously-rigid residues. Before fitting any residue,
`fit_order_parameters()` calibrates a single global scale factor on all
five τₖ (`model_free.calibrate_tau_scale`) against the presumed-rigid
subset (NOE above `noe_cut`), fit jointly on R1, R2, *and* NOE so the
correction doesn't leave a systematic NOE offset for the τₑ diagnostic to
trip on. `add_rigid_prediction()` uses the same calibration.

## Example

```python
from makeshift.relaxation import RelaxationProfile

prof = RelaxationProfile.from_bmrb(4390)     # eotaxin — single-field R1/R2/NOE
prof.fit_order_parameters()                  # auto structure + field from the entry
#   prof.fit_order_parameters(field_mhz=500)          # if the entry lacks the field tag
#   prof.fit_order_parameters(source="afdb")          # force a specific structure source

t = prof.table
t[["Seq_ID", "S2", "S2_err", "mf_model", "tau_e_ps", "Rex"]]
```

| Column | Meaning |
|---|---|
| `S2` / `S2_err` | fitted order parameter and its (approximate) standard error |
| `mf_model` | `"1"`, `"2"`, `"3"`, `"ambiguous"`, or `None` (no fit attempted) |
| `tau_e_ps` | internal correlation time in ps (Model 2 only) |
| `Rex` | exchange contribution to R2, s⁻¹ (Model 3 only) |
| `NOE_pred_rigid`, `noe_flag`, `r2_flag` | the diagnostic used for model selection |

`self.tau_scale` holds the calibrated diffusion-tensor timescale
correction after either `fit_order_parameters()` or
`add_rigid_prediction()` has run — call both back to back and they'll
agree, since they share the same calibration step.

```python
prof.add_rigid_prediction()      # same tau_scale as fit_order_parameters()
prof.fit_order_parameters()
print(prof.tau_scale)
```

## Scope

Fit only what a single field can support: Models 1–3, one rigid monomeric
solution-tumbling domain. Not fit: Models 4/5 (see above), multi-field
joint refinement (each `RelaxationProfile` is built at one resolved field —
see [`from_entry`](../api/relaxation.md)'s field resolution), and anything
that isn't a single rigid domain (multi-domain constructs with a flexible
linker, membrane/micelle-embedded peptides, obligate oligomers) — the
diffusion tensor these predictions are built on assumes one rigid body.

## Validation

Fitted S² was checked against deposited BMRB model-free S² across 29
single-field BMRB entries (pooled Pearson r ≈ 0.78); see the
[validation report](../order_parameters_validation.md) for the full
breakdown, what was excluded and why, and the bugs the process found.

## Full API

See [`makeshift.relaxation.model_free`](../api/relaxation.md) for the
underlying spectral-density/fitting functions, and the [Relaxation
reference](../api/relaxation.md) for `RelaxationProfile` itself.
