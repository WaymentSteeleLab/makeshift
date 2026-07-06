# CPMG dispersion

[`CPMGExperiment`](../api/cpmg.md) runs a full CPMG relaxation-dispersion
pipeline: it reads a set of `.ucsf` planes, fits peak lineshapes, computes
effective R₂ (R₂,eff) per peak across the νCPMG series, classifies which peaks
show exchange, and writes a results CSV (with optional plots).

!!! warning "nmrglue and NumPy 2.x"
    This module reads spectra via [`nmrglue`](https://nmrglue.readthedocs.io/).
    Pin `numpy<2` and `scipy<1.14` if you hit a `data type 'a8'` error. See
    [Installation](../installation.md).

## Running the pipeline

The experiment is configured from a YAML file, then run:

```python
from makeshift.relaxation import CPMGExperiment

exp = CPMGExperiment.from_config("SHP2_NSH2_CPMG.yaml")
exp.run(out_dir="out", make_plots=True)
```

There's also a thin functional wrapper if you prefer a one-liner:

```python
from makeshift.relaxation.cpmg import run_protein

run_protein("SHP2_NSH2_CPMG.yaml", out_dir="out")
```

## The config file

The YAML describes the constant-time delay, the reference plane, the sequence,
an optional structure for the rigid-body comparison, and one entry per νCPMG
plane. An abridged example (see `examples/SHP2_NSH2_CPMG.yaml`):

```yaml
time_T2: 0.04            # constant-time CPMG delay, seconds
data_dir: ~/.makeshift/datasets/SHP2_NSH2_CPMG/SHP2_NSH2_CPMG
reference: nuCPMG_0_1_SH2-WB_N15-CPMG.ucsf
peaklist: 52759          # a BMRB id, or a peak-list file
sequence: MTSRRWFHPNITGVEAENLLLTRGVDGSFLARPSKSNPGDF...
pdb: 9EHD
baseline_ref_plane: 50

planes:
  - file: nuCPMG_50_6_SH2-WB_N15-CPMG.ucsf
    vcpmg: 50
  - file: nuCPMG_100_9_SH2-WB_N15-CPMG.ucsf
    vcpmg: 100
  # ... one entry per plane ...

fit_lineshapes_cache: out/caches/SHP2_NSH2_CPMG_fit_lineshapes.json.zip
hydronmr_r2_cache: ../examples/SHP2_NSH2_CPMG_hydronmr_r2.csv
```

| Key | Meaning |
|---|---|
| `time_T2` | Constant-time CPMG delay (s) |
| `data_dir` | Directory holding the `.ucsf` planes |
| `reference` | Reference plane filename |
| `peaklist` | BMRB id (or file) providing assignments |
| `sequence` | One-letter sequence |
| `pdb` | Structure for the rigid-body R₂ comparison |
| `planes` | List of `{file, vcpmg}` for each dispersion point |
| `fit_lineshapes_cache` | Cache path for fitted lineshapes (speeds re-runs) |
| `hydronmr_r2_cache` | Cached HYDRONMR rigid R₂ |

The example dataset is downloadable — see [Datasets & structures](datasets.md):

```python
from makeshift.utils import datasets
datasets.fetch("SHP2_NSH2_CPMG")
```

## Run options

`run()` exposes the pipeline knobs: the `lineshape` model, an
`max_r2err_threshold` for filtering noisy fits, plotting toggles and axis
limits, and a `color_map` for the classification plots. See the
[`CPMGExperiment.run` reference](../api/cpmg.md) for the full signature.

## Outputs

`run()` writes a per-peak results CSV combining assignments, R₂,eff values, and
exchange classification to `out_dir`. With `make_plots=True` it also produces
per-peak dispersion curves, a grid overview, and a sequence waterfall.

## Full API

See the [CPMG reference](../api/cpmg.md).
