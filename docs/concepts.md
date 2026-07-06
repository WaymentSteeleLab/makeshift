# NMR-STAR concepts

`makeshift` reads [NMR-STAR](https://pynmrstar.readthedocs.io/en/latest/) files —
the deposition format used by the [BMRB](https://bmrb.io/). A little vocabulary
makes the API much easier to navigate.

## The building blocks

NMR-STAR files are organised around **saveframes**, each belonging to a
**category** (e.g. `assigned_chemical_shifts`, `entity`, `sample`). Within a
saveframe, tabular data lives in **loops**. The three concepts you interact with
most:

- **Entry** — a single BMRB deposition (one `.str` file). Represented by
  [`NMRStarEntry`](api/entry.md).
- **Entity** — a distinct molecular species (protein, DNA strand, ligand), each
  with its own `Entity_ID`. A complex may have several. Most methods accept an
  `entity_id=` argument to select one.
- **Chemical shift list** — the `_Atom_chem_shift` loop inside an
  `assigned_chemical_shifts` saveframe; one row per observed shift.

## How this maps to the API

| NMR-STAR idea | In makeshift |
|---|---|
| Whole entry | [`NMRStarEntry`](api/entry.md) |
| Saveframe categories | `entry.categories()` |
| A specific saveframe | `entry.saveframe(category, framecode=None)` |
| A loop as a table | `NMRStarEntry.loop_to_dataframe(loop)` |
| Any loop, flattened across saveframes | `entry.data_loop(category, loop_name, tags=None)` |
| Chemical shift loop | [`ChemicalShifts`](api/chemshift.md) |
| Entities & sequences | `entry.sequences()` / `entry.polymer_type()` |

## Escape hatches

Not every field in NMR-STAR has a dedicated method. When you need something the
convenience methods don't cover:

- `entry.categories()` returns an attribute-accessible mapping of every
  saveframe category present, so you can discover what an entry contains.
- `entry.data_loop(category, loop_name, tags=None)` flattens an arbitrary loop
  (from every matching saveframe) into a single DataFrame — for example
  `entry.data_loop("spectral_density_values", "_Spectral_density")`.

These let you reach any deposited data without leaving the tidy-DataFrame world.
