# RCI: flexibility from chemical shifts

`makeshift.rci` predicts per-residue backbone flexibility — the Random Coil Index
(Berjanskii & Wishart, *J. Am. Chem. Soc.* 2005) — from assigned backbone shifts.

The idea is that a residue whose shifts sit close to their random-coil values is
sampling many conformations, while a residue far from random coil is locked into
one. RCI turns the size of that deviation, smoothed along the chain, into a
single number per residue: **low RCI means ordered, high RCI means flexible.**

Unlike [TALOS-N](talosn.md), this runs in pure Python. There's no binary to
install and no network access needed once you have the shifts.

## Running it

```python
from makeshift.rci import RCI

r = RCI.from_bmrb(4403)
r.run()
r.results
```

`results` is a DataFrame with `Seq_ID`, `Comp_ID`, `RCI`, and `S2`.

If you already have a [`ChemicalShifts`](chemical-shifts.md) — say you
re-referenced it first — `calc()` builds and runs in one step:

```python
from makeshift import ChemicalShifts

cs = ChemicalShifts.from_bmrb(4403, reref="lacs", keep_download=True)
r = RCI.calc(cs)
```

You can also pass a shift table and sequence directly, which is what to do for
data that didn't come from BMRB:

```python
r = RCI(shifts, sequence=SEQ, first_resid=1).run()
```

!!! warning "Pass the sequence"
    The neighbor corrections need to know the residue on either side, including
    residues with no assigned shifts. Leave `sequence` out and it gets inferred
    from the shift table, corrections go missing wherever the chain has a gap,
    and you get a warning. `from_bmrb()` and `from_entry()` resolve the full
    polymer sequence for you.

## Which residue does the sequence start at?

Shift lists routinely start partway into a chain — a disordered N-terminus or a
cleaved tag simply has nothing assigned. Taking the first *assigned* residue as
the first residue of the sequence would then misalign every random-coil lookup
down the whole chain.

`from_entry()` and `from_bmrb()` avoid this by reading the entity's own
numbering out of `_Entity_comp_index` (see
[`NMRStarEntry.resolve_first_resid`](../api/entry.md)). BMRB 15490 is a good
example of why it matters: 181 residues, nothing assigned until residue 124.
When you construct `RCI` directly, `first_resid` is yours to set.

## Two algorithms

`algorithm=` picks the calculation:

| | `"wishart"` (default) | `"talosn"` |
|---|---|---|
| Ports | `rci_v_1c.py`, the reference script | the RCI-S² module inside TALOS-N |
| Gaps | filled from neighbors, up to 2 residues | none; strict ±1 window |
| Unobserved atoms | skipped | given a synthesized deviation |
| S² relation | `1 − 0.5·ln(1 + 10·RCI)` | `1.003 − 0.4·ln(1 + 17.7·RCI)` |
| `neighbor_table` | honored | ignored |

They agree closely on RCI itself (r ≈ 0.997 on the reference test case), so pick
`"wishart"` unless you specifically want to reproduce what TALOS-N would print.

!!! warning "The two S² columns are not on the same scale"
    Both backends emit a column called `S2`, but from different relations —
    roughly a 0.03–0.06 offset, in the same direction across the whole chain.
    Don't plot them on shared axes or difference them without accounting for
    that. See the [validation report](../rci_validation.md).

Two more things to know about `algorithm="talosn"`:

- It ignores `neighbor_table` — TALOS-N's tables are compiled into the binary.
- Residues with no usable data come back as `9999.0` in both `RCI` and `S2`,
  TALOS-N's no-data marker, passed through rather than converted to `NaN`.
  Filter with `results[results.RCI < 9999]` before averaging or plotting.

## Neighbor corrections

`neighbor_table` chooses which set of preceding/next-residue corrections to
apply to the random coil reference:

```python
r = RCI.from_bmrb(4403, neighbor_table="wang").run()
```

The options are `"schwarzinger"` (the reference script's own default), `"wang"`,
and `"schwartz_wang"`. Nothing in `makeshift` predicts secondary structure, so
all three use their coil-state values; the tables still differ from each other,
since each was fit independently.

## Comparing against TALOS-N

TALOS-N reports its own S², from a trained neural network rather than from RCI.
To compare the two:

```python
from makeshift import talosn
from makeshift.rci import RCI

tn = talosn.TalosN.from_bmrb(4403, data_dir=data_dir)
s2_ann = tn.predict_s2()                        # TALOS-N's ANN prediction
s2_rci = RCI.from_bmrb(4403).run().results      # shift-deviation prediction
```

Note that `TalosN.predict_s2()` and `RCI(algorithm="talosn")` are *not* the same
quantity: the first is the neural network output, the second reproduces the
separate RCI-S² calculation TALOS-N also carries out.

## Validation

Both backends are validated against their references —
`algorithm="wishart"` to machine precision against `rci_v_1c.py`'s own bundled
test case, `algorithm="talosn"` against the compiled binary across 9 BMRB
entries. The [RCI validation report](../rci_validation.md) has the details,
including three bugs that comparison surfaced.

## Full API

See the [RCI reference](../api/rci.md).
