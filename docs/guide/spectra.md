# Spectra

`makeshift.spectra` reads 2D Sparky `.ucsf` spectra, picks peaks, aligns peak
lists, and provides plotting helpers for spectra, peak lists, and chemical-shift
perturbations.

!!! warning "nmrglue and NumPy 2.x"
    This module imports [`nmrglue`](https://nmrglue.readthedocs.io/). Its current
    release is incompatible with NumPy 2.x — pin `numpy<2` and `scipy<1.14` if you
    hit a `data type 'a8'` error. See [Installation](../installation.md).

## Loading a spectrum

[`Spectrum`](../api/spectra.md) wraps the intensity array plus one nmrglue
unit-conversion object per axis (for ppm ↔ point conversions).

```python
from makeshift.spectra import Spectrum

spec = Spectrum.from_ucsf("hsqc.ucsf")
spec.data           # intensity array (axis 0 = indirect/N, axis 1 = direct/H)
spec.uc             # per-axis unit-conversion objects
```

## Picking peaks

```python
# Noise-floor estimate for setting a sensible baseline
noise = spec.estimate_background()

# Pick peaks in the amide region of a 1H-15N spectrum
peaks = spec.pick_peaks(baseline=noise)

# Convert a point index to ppm on a given axis
ppm = spec.ppm(axis=1, point=512)
```

`pick_peaks` accepts an explicit `baseline`, a peak-picking `algorithm`, and
`h_ppm_min` / `h_ppm_max` bounds to restrict the amide window.

## Aligning peak lists

[`map_peaklists`](../api/spectra.md) aligns two peak lists in (H_ppm, N_ppm) and
matches them one-to-one. The `right` list (e.g. a reference assignment) is
shifted by a translation offset — grid-searched if you don't supply one — then
Hungarian-matched to the fixed `left` list within tolerance:

```python
from makeshift.spectra import map_peaklists

left_out, right_mapped = map_peaklists(
    left_peaks.data,          # fixed (e.g. picked from a spectrum)
    right_peaks.data,         # shifted onto left (e.g. reference assignment)
    offset=None,              # (ΔH, ΔN); grid-searched if None
    tol=1.0,                  # multiplier on default (0.03, 0.3) ppm tolerances
    how="inner",
)
```

`left_out` carries the transferred labels plus a `conflict` column flagging
right-side peaks that lost a close match to a competitor; `right_mapped` is the
full right list after the offset, ready for plotting.

## Plotting

Three helpers, all of which accept an existing Matplotlib `ax` so you can layer
them:

```python
from makeshift.spectra.plotting import plot_spectrum, plot_peaklist, plot_csp

ax = plot_spectrum(spec.data)                 # contour plot
plot_peaklist(peaks.data, ax=ax)              # markers + optional labels
plot_csp(peaks1, peaks2, on="label", ax=ax)   # matched pairs joined by lines
```

`plot_csp` draws connecting lines between paired peaks in two matched lists — the
standard way to visualise chemical-shift perturbations across a titration or
mutation.

## Full API

See the [Spectra reference](../api/spectra.md).
