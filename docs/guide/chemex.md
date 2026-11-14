# ChemEx exchange fitting

`makeshift.chemex` wraps [ChemEx](https://github.com/gbouvignies/ChemEx)
(Guillaume Bouvignies) to fit CPMG relaxation-dispersion profiles to a
chemical-exchange model. For a two-state model, it returns the exchange rate
k<sub>ex</sub>, the minor-state population p<sub>B</sub>, and per-residue
chemical-shift differences Δω and intrinsic R₂.

It picks up where the [CPMG pipeline](cpmg.md) leaves off. `CPMGExperiment`
fits peak volumes in every νCPMG plane, and [`ChemEx`](../api/chemex.md) turns
those volumes into ChemEx input files, runs `chemex fit` and parses the results
back into DataFrames. You never have to write a ChemEx TOML by hand.

## Install ChemEx

ChemEx is not a makeshift dependency. Install it in the same environment:

```bash
pip install chemex
```

ChemEx has no stable Python API, so the wrapper runs it on the command line
(`python -m chemex fit ...`) using the current interpreter.

## Fitting a CPMG experiment

Run the CPMG pipeline first, then hand the experiment to ChemEx:

```python
from makeshift.relaxation import CPMGExperiment

exp = CPMGExperiment.from_config("SHP2_NSH2_CPMG.yaml")
exp.run(out_dir="out")

cx = exp.to_chemex(pw90=35e-6, temperature=25.0)
cx.run(output_dir="out/chemex")

cx.global_parameters      # kex, pB (+ derived pA, kAB, kBA), with errors
cx.residue_parameters     # per-residue DW_AB, R2_A (+ _err), with ref_index
cx.fit_profiles           # experimental vs fitted intensity and R2eff per νCPMG
cx.statistics             # chi-square, reduced chi-square, AIC, BIC
```

`to_chemex` is shorthand for `ChemEx.from_cpmg(exp, ...)`. It reads the ¹H
field and the ¹⁵N carrier from the reference spectrum's `.ucsf` header. The
15N 90° pulse width `pw90` (s) and the `temperature` (°C) aren't in the
header, so you always supply them. Pass `h_larmor_frq=` (MHz) or `carrier=`
(¹⁵N, ppm) to override the header values.

Only peaks with an assignment (`assn_label`, e.g. `Y4`) are fitted. ChemEx
needs residue names such as `Y4N-H` to tell spins apart.

### From lineshape fits directly

Without a `CPMGExperiment`, build from
[`fit_peaks`](../api/relaxation.md) output plus a peak table:

```python
from makeshift.chemex import ChemEx

cx = ChemEx.from_fit_results(
    fit_results, vcpmg_values, time_T2, ref_df,
    pw90=35e-6, temperature=25.0, h_larmor_frq=600.0, carrier=117.3,
)
cx.run()
```

`ref_df` needs `ref_index`, `assn_label` and `N_ppm` columns.

## What gets fitted

By default the wrapper fits the two-state model (`model="2st"`), floating
`PB`, `KEX_AB`, `DW_AB` and `R2_A`. Starting values are p<sub>B</sub> = 0.05
(bounded to 0–0.5 so state A stays the major state), k<sub>ex</sub> = 500 s⁻¹,
and Δω = 1 ppm for every residue.

Each residue's ground-state ¹⁵N shift (`CS_A`) is set from its peak position
(`N_ppm`). This matters: ChemEx's default is 0 ppm, which puts every residue
~118 ppm off resonance and badly biases p<sub>B</sub>.

Change starting values with `parameters=` and the fitted set with `method=`:

```python
cx = exp.to_chemex(
    pw90=35e-6, temperature=25.0,
    parameters={"GLOBAL": {"KEX_AB": 1500.0}, "DW_AB": 2.0},  # scalar = every residue
    method=["PB", "KEX_AB", "DW_AB", "R2_A"],
)
```

- **`parameters`** is keyed by ChemEx section. Values may be
  `[value, min, max]`.
- **`method`** accepts a list of parameter names (one fitting step), a dict of
  ChemEx v2 method steps, or the path to an existing method file. For example,
  to fit R₂ alone first and then everything:

```python
method = {
    "STEP1": {"ROLES": [{"FIT": ["R2_A"]}]},
    "STEP2": {"ROLES": [{"FIT": ["PB", "KEX_AB", "DW_AB", "R2_A"]}]},
}
```

For a multi-step method, results come from the last step. Every step is kept
in `cx.results["steps"]`.

Other experiments and models work through the same object. Set
`experiment="cpmg_15n_ip"` (or any ChemEx experiment name) and `model=`, and
pass `method=` for models without a default. Use `experiment_settings=` for
extra `[experiment]` keys such as `{"antitrosy": True}`.

## Run options

| Argument | Meaning |
|---|---|
| `output_dir` | Where to write `inputs/` and ChemEx's `chemex/` output. Omit for a temporary directory. |
| `plot` | ChemEx plotting level: `"nothing"` (default), `"normal"`, `"all"`. PDFs are listed in `results["plots"]`. |
| `include` / `exclude` | Spin names (`Y4N`) to restrict the fit to or leave out. |
| `workers` | ChemEx worker processes. |
| `cleanup` | Delete the temporary directory after parsing. It is kept if ChemEx fails, so you can inspect the inputs. |

If ChemEx exits with an error, `run()` raises a `RuntimeError` that includes
ChemEx's own message and the path to the inputs it was given.

## Running ChemEx by hand

To prepare files for a ChemEx run of your own instead, write the profiles with
`save_chemex`. It writes one `.out` file per assigned peak plus a
`[data.profiles]` snippet to paste into an experiment file:

```python
exp.save_chemex("chemex_profiles/")
```

## Full API

See the [ChemEx reference](../api/chemex.md).
