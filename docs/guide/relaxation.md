# Relaxation & dynamics

Two layers here: pull deposited relaxation straight out of an entry with
[`NMRStarEntry`](../api/entry.md), or assemble it into a per-residue dynamics
profile with [`RelaxationProfile`](../api/relaxation.md), in the style of RelaxDB
([Wayment-Steele, El Nesr et al.](https://www.biorxiv.org/content/10.1101/2025.03.19.642801)).

## Deposited data, straight from an entry

```python
import makeshift as ms

entry = ms.NMRStarEntry.from_bmrb(25013)
entry.datasets()                # which data types the entry holds
entry.relaxation("T2")          # R2 — also "T1"/"R1", "T1rho", "NOE"; units-aware
entry.order_parameters()        # model-free S2 (S2, Tau_e, Rex)

# Anything without a dedicated method:
entry.data_loop("spectral_density_values", "_Spectral_density")
```

## Building a dynamics profile

[`RelaxationProfile`](../api/relaxation.md) aligns R1/R2/NOE to the sequence,
forms the R₂/R₁ observable, compares it to a HYDRONMR rigid-body prediction, and
labels each residue by motional regime.

```python
from makeshift.relaxation import RelaxationProfile

prof = RelaxationProfile.from_bmrb(25013)   # pulls T1/T2/NOE, aligns to sequence
prof.add_rigid_prediction()                 # runs HYDRONMR on a structure
print(prof.label())                         # per-residue motion string
prof.plot("R2_R1")
```

You can also build from an already-parsed entry with
`RelaxationProfile.from_entry(entry)`.

### The table

`prof.table` has one row per sequence position (1-indexed), with columns
`Seq_ID`, `residue`, `R1`, `R1_err`, `R2`, `R2_err`, `NOE`, `NOE_err`, `R2_R1`,
`R2_R1_err`, and `has_data` — plus `scaled_R2_R1_pred` and `label` once the
rigid-prediction and labelling steps have run.

## The rigid-body prediction

`add_rigid_prediction()` runs [HYDRONMR](hydronmr.md) on a structure and scales
its rigid R₂/R₁ to the data. The structure can be:

- a **local PDB file** — `add_rigid_prediction("model.pdb")`
- a **PDB id** (fetched from RCSB) — `add_rigid_prediction("1WRP")`
- a **UniProt accession** (fetched from AlphaFold DB) — `add_rigid_prediction("P0DP23")`

With no argument it uses the first available PDB or AlphaFold model based on the entry cited.

> [!NOTE]
> `makeshift` does **not** predict structure itself — it only fetches an
> existing experimental or predicted model. See
> [Datasets & structures](datasets.md) for more information or provide your own prediction.

## Motion labels

Labels for each residue by motion regime against a HYDRONMR rigid-body
prediction — this follows the RelaxDB curation described in (Wayment-Steele, El Nesr et al)(https://www.biorxiv.org/content/10.1101/2025.03.19.642801v3).

`label()` assigns one token per residue and returns the label string:

| Token | Meaning |
|:---:|---|
| `A` | ordered |
| `^` | µs–ms exchange (elevated R₂/R₁) |
| `v` | ps–ns motion (hetNOE ≤ 0.65) |
| `b` | both |
| `.` | peak missing |
| `t` | disordered terminus |
| `p` | proline |

## Plotting

```python
prof.plot("R2_R1")   # any observable column, along the sequence
```

## Full API

See the [Relaxation reference](../api/relaxation.md). The CPMG
relaxation-dispersion pipeline has its [own guide](cpmg.md).
