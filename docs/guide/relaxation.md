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
labels each residue by motional regime. Pass a
[`PeakList`](peaklists.md) for the assigned residues so positions without a peak
are marked `.` rather than mistaken for dynamics.

```python
from makeshift import PeakList
from makeshift.relaxation import RelaxationProfile

pl = PeakList.from_bmrb(19151)
prof = RelaxationProfile.from_bmrb(25013, peaklist=pl)  # T1/T2/NOE, aligned to sequence
#    peaklist= can also be a residue-id list, e.g. [5, 6, 7]
prof.add_rigid_prediction()                              # auto structure from the entry
#    prof.add_rigid_prediction(source="rcsb")            # force deposited PDB
#    prof.add_rigid_prediction(source="afdb")            # force AlphaFold / UniProt
#    prof.add_rigid_prediction("1WRP")                   # specific PDB id
labels = prof.label(rex_n_std=1.0, noe_cut=0.65)
print(labels)                                            # e.g. AAAA^AA..vAb…p
prof.plot("R2_R1")
```

You can also build from an already-parsed entry with
`RelaxationProfile.from_entry(entry, peaklist=...)`.

### The table

`prof.table` has one row per sequence position (1-indexed), with columns
`Seq_ID`, `residue`, `R1`, `R1_err`, `R2`, `R2_err`, `NOE`, `NOE_err`, `R2_R1`,
`R2_R1_err`, and `has_data` — plus `scaled_R2_R1_pred` and `label` once the
rigid-prediction and labelling steps have run.

## The rigid-body prediction

`add_rigid_prediction()` runs [HYDRONMR](hydronmr.md) on a structure and scales
its rigid R₂/R₁ to the data.

### Choosing a structure

Pass a structure explicitly, or let makeshift pick one from the entry:

```python
prof.add_rigid_prediction()                    # auto: entry PDB → else AlphaFold
prof.add_rigid_prediction(source="rcsb")       # force the entry's deposited PDB
prof.add_rigid_prediction(source="afdb")       # force the entry's AlphaFold/UniProt model
prof.add_rigid_prediction("1WRP")              # a specific PDB id (RCSB)
prof.add_rigid_prediction("P0DP23")            # a UniProt accession (AlphaFold DB)
prof.add_rigid_prediction("my_model.pdb")      # a local file
```

| `source` | When `pdb=` is omitted | When `pdb=` is given |
|---|---|---|
| `"auto"` (default) | entry's PDB if cited, else its AlphaFold/UniProt id | infer from the identifier (path / 4-char PDB / UniProt) |
| `"rcsb"` | first PDB id cited by the entry | fetch that id from RCSB |
| `"afdb"` | first AlphaFold/UniProt accession cited by the entry | fetch that accession from AlphaFold DB |
| `"file"` | — | treat `pdb` as a local path |

If the entry cites neither a PDB nor an AlphaFold model and you pass nothing,
this raises — makeshift does **not** predict structure. See
[Datasets & structures](datasets.md) for `fetch_structure` details.

## Motion labels

Labels for each residue by motion regime against a HYDRONMR rigid-body
prediction — this follows the RelaxDB curation described in (Wayment-Steele, El Nesr et al)(https://www.biorxiv.org/content/10.1101/2025.03.19.642801v3).

`label()` assigns one token per residue and returns the label string. Call it
**after** `add_rigid_prediction()` (and pass a peaklist when building the
profile) before plotting.

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
