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

## Settings in the CPMG config

ChemEx needs the spectrometer field, ¹⁵N carrier, temperature, ¹⁵N 90° pulse
width and CPMG delays. The easiest way to supply most of them is to point the
CPMG config at the experiment's Bruker `acqus` file:

```yaml
data_dir: ~/data/my_protein
acqus: acqus              # relative to data_dir, or an absolute path
time_T2: 0.04             # CPMG constant-time period T (s): always required
reference: ref.ucsf
planes:
  - {file: p50.ucsf, vcpmg: 50}
  # ...
```

With `acqus`, no `chemex:` section is needed. These values are read from it:

| Setting | From acqus | Unit |
|---|---|---|
| `time_equil` | `d20` | s |
| `pw90` | `p30 / 2` (p30 is the ¹⁵N CPMG 180°) | s |
| `carrier` | `O3 / BF3` | ¹⁵N ppm |
| `h_larmor_frq` | `BF1` | MHz |
| `temperature` | `TE` (K → °C) | °C |

!!! warning "Check the delay and pulse indices"
    Which `d` and `p` index holds which quantity depends on the pulse
    program. The mapping above is for N15CPMG.wtg. `load_config` prints the
    pulse program and the values it read, so compare them with your
    experiment, especially when using a different pulse program.

`time_T2` is never read from acqus. How the constant-time period relates
to the delays depends on the pulse program, and in N15CPMG.wtg.NEW_tst1, d17
is not T. Take T from the pulse program or your lab notebook. `load_config`
prints d17 next to `time_T2` for comparison.

Every νCPMG × `time_T2` must be a whole number of CPMG cycles. If it isn't,
building the ChemEx fit raises a `ValueError` listing the offending planes.
For example, `time_T2 = 0.03` with 50 Hz steps gives 1.5 cycles. A wrong T
otherwise produces a jagged ChemEx fit curve that dips at every plane with a
half cycle.

### The optional `chemex:` section

Use a `chemex:` section to choose the ChemEx experiment and model, to
override anything read from acqus, or to supply the values yourself when you
have no acqus file:

```yaml
chemex:
  experiment: cpmg_15n_tr   # ChemEx experiment name (default cpmg_15n_ip)
  temperature: 27.0         # overrides TE from acqus
```

| Key | Meaning | Default |
|---|---|---|
| `experiment` | ChemEx experiment name, e.g. `cpmg_15n_ip`, `cpmg_15n_tr` | `cpmg_15n_ip` |
| `h_larmor_frq` | ¹H Larmor frequency (MHz) | acqus, else the reference `.ucsf` header |
| `carrier` | ¹⁵N carrier (ppm) | acqus, else the reference `.ucsf` header |
| `temperature` | Sample temperature (°C) | acqus; **required** without it |
| `pw90` | ¹⁵N 90° pulse width (s) | acqus; **required** without it |
| `model` | ChemEx kinetic model: `2st`, `1st`, `3st`, ... | `2st` |
| `experiment_settings` | Any other ChemEx `[experiment]` key, e.g. `{antitrosy: true}`; merged with `time_equil` from acqus | — |
| `conditions` | Any other ChemEx `[conditions]` key, e.g. `{p_total: 1.0e-3}` | — |
| `parameters` | Starting values, see [What gets fitted](#what-gets-fitted) | — |
| `method` | What to fit, see [What gets fitted](#what-gets-fitted) | — |

The acqus file doesn't say which ChemEx experiment a pulse program
corresponds to, so set `experiment` yourself if it isn't `cpmg_15n_ip`
(for example `cpmg_15n_tr` for the TROSY version).

The settings are checked when the YAML is loaded. An unknown key (such as the
typo `pw_90`), a missing `temperature` or `pw90` (with no acqus), or a
non-numeric value raises a `ValueError` straight away, before any lineshape
fitting.

## Fitting a CPMG experiment

Run the CPMG pipeline first, then hand the experiment to ChemEx:

```python
from makeshift.relaxation import CPMGExperiment

exp = CPMGExperiment.from_config("my_protein_cpmg.yaml")
exp.run(out_dir="out")

cx = exp.to_chemex()          # settings from acqus and/or the chemex: section
cx.run(residues=["Y4", "G10", "L22"], fit="global")

cx.global_parameters      # shared kex, pB (+ derived pA, kAB, kBA), with errors
cx.residue_parameters     # per-residue DW_AB, R2_A (+ _err), with ref_index
cx.fit_profiles           # experimental vs fitted intensity and R2eff per νCPMG
cx.statistics             # chi-square, reduced chi-square, AIC, BIC
```

`to_chemex` is shorthand for `ChemEx.from_cpmg(exp)`. Without acqus, a
missing field or carrier is read from the reference spectrum's header, and
the value used is printed.

To try a variation without editing the YAML, pass any setting as a keyword.
It replaces just that entry:

```python
cx = exp.to_chemex(model="1st")
```

Only peaks with an assignment (`assn_label`, e.g. `Y4`) are fitted. ChemEx
needs residue names such as `Y4N-H` to tell spins apart.

### From lineshape fits directly

If you call [`fit_peaks`](../api/relaxation.md) yourself instead of
`CPMGExperiment.run()`, pass its output, the assigned peak table and the
config:

```python
from makeshift.chemex import ChemEx
from makeshift.relaxation import load_config, load_planes, fit_peaks

config = load_config("my_protein_cpmg.yaml")
ref_spectrum, plane_data, vcpmg_values, time_T2 = load_planes(config)
# ... pick peaks and assign them -> `assigned` ...
fit_results = fit_peaks(plane_data, assigned, ref_spectrum.uc, config, lineshape="gaussian")

cx = ChemEx.from_fit_results(fit_results, assigned, config, ref_spectrum=ref_spectrum)
cx.run(residues=["Y4", "G10"], fit="global")
```

`time_T2`, the νCPMG values and the ChemEx settings all come from `config`
(which can also be the YAML path). `ref_spectrum` is optional. It is only
used for the field and carrier when there is no acqus file, and if you
leave it out the reference `.ucsf` is read instead. `assigned` needs
`ref_index`, `assn_label` and `N_ppm` columns.

## Choosing residues: global or individual fits

`run()` needs only two choices:

- **`residues`**: which residues to fit. Write them as labels (`"Y4"`), spin
  names (`"Y4N"`), profile names (`"Y4N-H"`) or residue numbers (`4`). Leave it
  out to fit every assigned residue. A residue with no assigned peak raises a
  `ValueError` listing the ones that are available.
- **`fit`**: how kex and pB are shared.

| `fit=` | What happens | Results |
|---|---|---|
| `"global"` (default) | One ChemEx fit in which every selected residue shares k<sub>ex</sub> and p<sub>B</sub>; Δω and R₂ are per residue. | `global_parameters` holds k<sub>ex</sub> and p<sub>B</sub>; `residue_parameters` holds Δω and R₂. |
| `"individual"` | ChemEx is run once per residue, so each residue gets its own k<sub>ex</sub> and p<sub>B</sub>. | `residue_parameters` holds `KEX_AB`, `PB`, `DW_AB`, `R2_A` (and errors) for each residue; `global_parameters` is None; `statistics` has one row per residue. |

```python
cx.run(residues=["Y4", "G10", "L22"], fit="individual")
cx.residue_parameters[["KEX_AB", "KEX_AB_err", "PB", "PB_err"]]
```

### Workflow: individual fits, then a global fit

The usual hands-on route is to fit residues individually, check whether
they agree on k<sub>ex</sub> and p<sub>B</sub>, and then fit them globally
starting from the consensus values.

```python
# 1. choose residues, e.g. the peaks classified as exchanging ("Rex")
residues = exp.classifications.query("label == 'Rex'")["assn_label"].dropna().tolist()
# or by hand: residues = ["Y4", "G10", "L22"]

# 2. fit each residue on its own
cx.run(residues=residues, fit="individual")
cx.residue_parameters[["KEX_AB", "KEX_AB_err", "PB", "PB_err"]]
cx.exchange_summary()

# 3. global fit, starting from the values you settled on
s = cx.exchange_summary()
cx.run(residues=residues, fit="global",
       kex=s.loc["kex", "mean"], pb=s.loc["pb", "mean"])
cx.global_parameters
```

With the lower-level functions, use the `classifications` table returned by
`classify_peaks` in place of `exp.classifications`.

`exchange_summary()` works after an individual fit and has one row each for
`kex` and `pb`:

| Column | Meaning |
|---|---|
| `mean`, `mean_err` | Error-weighted mean (weights 1/err²) and its standard error |
| `median`, `min`, `max` | Unweighted spread |
| `chi2_dof` | Σ((x − mean)/err)² / (n − 1). Near 1: the residues agree within their errors. Much larger: they don't, and they may not share one exchange process. |
| `n` | Residues used (those with a finite, non-zero error) |

`kex` and `pb` in step 3 are starting values; the global fit still fits
them. Residues with small Δω are often poorly determined on their own (large
`KEX_AB_err`). The weighted mean down-weights them, and the global fit pools
their information. Drop residues whose individual values clearly disagree
before the global fit. The starting values must lie inside `kex_bounds` and
`pb_bounds`, or `run()` raises a `ValueError`.

## Checking the fits

Pass `plot=True` to `run()`, or call `cx.plot()` any time after a run, to see
the fits inline in the notebook. There is one panel per residue, in the same
layout as ChemEx's own plots:

- **Bottom:** R₂,eff vs νCPMG, with the data and its error bars and the
  ChemEx fit through the measured points.
- **Top:** residuals (data − fit) with error bars. A good fit scatters
  around zero; a systematic trend means the model or the inputs (time_T2,
  pw90, carrier) are wrong.
- **Panel titles:** the residue's fitted values. Individual fits show
  k<sub>ex</sub>, p<sub>B</sub>, Δω and R₂A. Global fits show Δω and R₂A, with
  the shared k<sub>ex</sub> and p<sub>B</sub> in the figure title.

```python
cx.run(residues=residues, fit="individual", plot=True)

cx.plot(residues=["Y92", "N87"])      # a subset
fig = cx.plot(show=False)             # get the figure instead, e.g. to save it
fig.savefig("fits.png", dpi=150)
```

`ncols=` sets the panels per row (default 4). ChemEx's own PDF plots are
switched off.

## What gets fitted

By default the wrapper fits the two-state model (`model: 2st`) with this
method:

```toml
[STEP1]
FIT = ["PB", "KEX_AB", "DW_AB"]
```

It's written in ChemEx's current method format
(`ROLES = [{ FIT = [...] }]`). `R2_A` isn't listed because ChemEx always fits
it for CPMG experiments, so the fitted set is `PB`, `KEX_AB`, `DW_AB` and
`R2_A`. `CS_A` stays fixed at each peak's ¹⁵N shift. Δω starts at 1 ppm for
every residue.

### k<sub>ex</sub> and p<sub>B</sub>: starting values and bounds

Set these as arguments to `run()`:

```python
cx.run(residues=["Y4", "G10"], fit="global",
       kex=1500.0, kex_bounds=(100, 1e4),    # s⁻¹
       pb=0.02, pb_bounds=(0.0, 0.1))
```

| Argument | Default | Meaning |
|---|---|---|
| `kex` | `500.0` | Starting k<sub>ex</sub> (s⁻¹) |
| `kex_bounds` | `(10, 1e4)` | `(min, max)` for k<sub>ex</sub> (s⁻¹) |
| `pb` | `0.05` | Starting p<sub>B</sub> |
| `pb_bounds` | `(0, 0.2)` | `(min, max)` for p<sub>B</sub>, within [0, 1]. The default allows a minor state of at most 20%. |

`None` on either side of a bounds tuple, or `None` for the whole tuple,
leaves that side unbounded. A start value outside its bounds raises a
`ValueError`. For individual fits every residue starts from the same values.
The values used are recorded in `cx.results["initial"]`.

Because ChemEx makes one local fit from these values, it's worth rerunning
with a different `kex` start (e.g. 200 and 2000) to check that the fit lands
in the same minimum.

k<sub>ex</sub> and p<sub>B</sub> can't be set in the YAML or in
`parameters`. Doing so raises an error pointing you to these arguments, so
there's never a question about which value was used.

### The ground-state shift `CS_A`

Each residue's ground-state ¹⁵N shift (`CS_A`) is fixed at its peak position
(`N_ppm`) by default. To let it float, use a method that fits it. It then
floats within ±`cs_window` ppm of the peak shift (default 0.5):

```yaml
chemex:
  method: [PB, KEX_AB, CS_A, DW_AB]
```

```python
cx.run(residues=["Y4", "G10"], cs_window=1.0)    # ±1 ppm, if the method fits CS_A
```

Using the peak position matters: ChemEx's default is 0 ppm, which
puts every residue ~118 ppm off resonance and badly biases p<sub>B</sub>.

!!! note "CPMG data barely constrains `CS_A`"
    `CS_A` only enters CPMG profiles through small off-resonance effects, so
    the fit learns little about it. Its reported errors are large, and
    floating it widens the k<sub>ex</sub> uncertainty somewhat. On simulated
    data, ±146 s⁻¹ with `CS_A` fixed became about ±260 s⁻¹ floating within
    ±0.5 ppm, with the same k<sub>ex</sub> and p<sub>B</sub>. Without bounds
    (`cs_window=None`) the fit usually fails to converge. That's why the
    default keeps it fixed.

### Separate R₂ for the minor state

By default ChemEx holds the minor state's intrinsic R₂ equal to the major
state's (`R2_B = R2_A`), so each residue has one R₂. Pass `separate_r2=True`
to fit `R2_B` on its own. That's useful when the two states should relax
differently, e.g. a structured major state and a more flexible minor state:

```python
cx.run(residues=residues, fit="global", kex=800, pb=0.05, separate_r2=True, plot=True)
cx.residue_parameters[["R2_A", "R2_A_err", "R2_B", "R2_B_err"]]
```

`separate_r2=True` adds `R2_B` to the method's `FIT` list (the last step of
a multi-step method). Panel titles then show R₂B next to R₂A.

CPMG data constrain `R2_B` weakly, because the minor state contributes only
about p<sub>B</sub> of the signal. Expect large errors on `R2_B`, or none at
all when it hits a bound. Expect k<sub>ex</sub> and p<sub>B</sub> to become
less certain too, because a higher `R2_B` can partly stand in for a larger
Δω or p<sub>B</sub>. Compare the fit with and without it (`cx.statistics`,
chi²/dof) before reading much into the R2_A vs R2_B difference. Global fits
help, since k<sub>ex</sub> and p<sub>B</sub> are then shared across residues.
With a method file, list `R2_B` under `FIT` in the file instead.

### Other parameters and the method

Change other starting values with `parameters` and the fitted set with
`method`, in the `chemex:` section:

```yaml
chemex:
  parameters:
    DW_AB: 2.0              # a single number applies to every residue
    R2_A: [12.0, 0.0, 50.0] # [start, min, max]
  method: [PB, KEX_AB, CS_A, DW_AB]   # e.g. also float CS_A
```

- **`parameters`** is keyed by ChemEx section (except k<sub>ex</sub> and
  p<sub>B</sub>, see above). Values may be `[value, min, max]`.
- **`method`** accepts a list of parameter names (one fitting step), a mapping
  of ChemEx v2 method steps, or the path to an existing method file. For
  example, to fit R₂ alone first and then everything:

```yaml
  method:
    STEP1: {ROLES: [{FIT: [R2_A]}]}
    STEP2: {ROLES: [{FIT: [PB, KEX_AB, DW_AB]}]}
```

For a multi-step method, results come from the last step. Every step is kept
in `cx.results["steps"]`.

Other experiments and models use the same section: set `experiment` and
`model`, and give a `method` for models without a default (anything other
than `2st` and `1st`).

## Run options

| Argument | Meaning |
|---|---|
| `residues` | Residues to fit (see above). Default: all assigned residues. |
| `fit` | `"global"` (default) or `"individual"`. |
| `cs_window` | If the method fits `CS_A`, it floats within ± this many ppm of each peak shift (default 0.5; None: unbounded). Unused with the default method. |
| `separate_r2` | Fit the minor state's R₂ (`R2_B`) separately from `R2_A` instead of holding them equal (see [above](#separate-r2-for-the-minor-state)). Default False. |
| `kex`, `kex_bounds`, `pb`, `pb_bounds` | Starting values and `(min, max)` bounds for k<sub>ex</sub> and p<sub>B</sub> (see [above](#kex-and-pb-starting-values-and-bounds)). |
| `output_dir` | Where to write ChemEx's `inputs/` and `chemex/` output (one sub-folder per residue for individual fits). Omit for a temporary directory. |
| `plot` | `True` to show the fitted profiles inline with matplotlib when the fit finishes (see [Checking the fits](#checking-the-fits)). Default `False`. |
| `cleanup` | Delete the temporary directory after parsing. It is kept if ChemEx fails, so you can inspect the inputs. |

If ChemEx exits with an error, `run()` raises a `RuntimeError` that includes
ChemEx's own message and the path to the inputs it was given.

## Running ChemEx by hand

To prepare files for a ChemEx run of your own instead, write the profiles with
`save_chemex`. It writes one `.out` file per assigned peak plus a
`[data.profiles]` snippet to paste into an experiment file:

```python
from makeshift.chemex import save_chemex

save_chemex(exp.fit_results, exp.vcpmg_values, exp.time_T2, "chemex_profiles/",
            ref_df=exp.ref_df)
```

## Full API

See the [ChemEx reference](../api/chemex.md).
