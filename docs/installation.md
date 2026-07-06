# Installation

## Requirements

`makeshift` requires **Python ≥ 3.8**. Its runtime dependencies are all
mainstream scientific-Python packages: `numpy`, `pandas`, `scipy`,
`scikit-learn`, `matplotlib`, `seaborn`, `nmrglue`, and `tqdm`.

## Install from GitHub

```bash
pip install git+https://github.com/WaymentSteeleLab/makeshift.git
```

## Install for local development

```bash
git clone https://github.com/WaymentSteeleLab/makeshift.git
cd makeshift
pip install -e .
```

## Optional extras

The demo notebooks need Jupyter:

```bash
pip install "makeshift[demos]"   # adds jupyter
```

!!! note "NumPy 2.x and nmrglue"
    The `makeshift.spectra` and `makeshift.relaxation` modules import
    [`nmrglue`](https://nmrglue.readthedocs.io/), whose current release (0.11)
    is **not compatible with NumPy 2.x** (`TypeError: data type 'a8' not
    understood`). If you hit this, pin an older NumPy alongside a compatible
    SciPy:

    ```bash
    pip install "numpy<2" "scipy<1.14"
    ```

    The core modules (`NMRStarEntry`, `ChemicalShifts`, `PeakList`) do **not**
    import nmrglue and work fine on NumPy 2.x.

## Component-specific downloads

Two subsystems fetch external assets on demand rather than bundling them:

- **TALOS-N** — the NIH binary and its database are downloaded into a specified 
  directory. See [TALOS-N prediction](guide/talosn.md).
- **Structures & datasets** — Use PDB or AlphaFoldDB structure and example datasets are
  fetched and cached under `~/.makeshift/`. See
  [Datasets & structures](guide/datasets.md).
