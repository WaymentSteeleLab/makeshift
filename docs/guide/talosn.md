# TALOS-N prediction

`makeshift.talosn` wraps the NIH
[TALOS-N](https://spin.niddk.nih.gov/bax-apps/software/TALOS-N/) binary (Shen &
Bax, *J. Biomol. NMR* 2013), which predicts backbone φ/ψ torsion angles,
per-residue S² order parameters, and secondary structure from assigned backbone
chemical shifts using a trained neural network.

## The binary is downloaded on demand

TALOS-N and its database aren't bundled — they're downloaded from NIH (under
their [Terms of Use](https://spin.niddk.nih.gov/bax-apps/terms.html), which the
installer prints) into a `data_dir` you choose.

!!! tip "Keep one data_dir"
    Store the path in a variable and pass the **same** `data_dir` to the
    installer and to every [`TalosN`](../api/talosn.md). If you omit it, it
    defaults to inside the installed package — usually not what you want for a
    few-hundred-MB download.

```python
from pathlib import Path
from makeshift import talosn

data_dir = Path.home() / "talosn_data"
talosn.install_talosn_data(data_dir=data_dir)     # one-time, ~ a few hundred MB
```

You can check installation status with `talosn.is_talosn_data_installed(data_dir)`.

## Running a prediction

```python
tn = talosn.TalosN.from_bmrb(4527, data_dir=data_dir)
tn.run()                    # or run(auto_install=True) to fetch on first use

tn.order_parameters         # predS2.tab — per-residue S2
tn.torsion_angles           # pred.tab   — phi/psi per residue + confidence class
tn.secondary_structure      # predSS.tab — helix / sheet / coil
```

Build from an already-parsed entry instead with
`TalosN.from_entry(entry, data_dir=data_dir)`. `predict_s2()` runs the pipeline
if needed and returns the S² table directly.

## Outputs

| Property | TALOS-N file | Contents |
|---|---|---|
| `torsion_angles` | `pred.tab` | φ/ψ per residue + a prediction confidence class |
| `order_parameters` | `predS2.tab` | per-residue S² |
| `secondary_structure` | `predSS.tab` | helix / sheet / coil probabilities |

## Terms of use

The TALOS-N software is distributed separately by NIH under its own
[Terms of Use](https://spin.niddk.nih.gov/bax-apps/terms.html) (including no
redistribution without permission from the authors). Those terms govern the
downloaded binary, not this wrapper.

## Full API

See the [TALOS-N reference](../api/talosn.md).
