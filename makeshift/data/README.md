# Reference data

CSV tables loaded by `makeshift/data/tables.py` and cached as module-level
dicts. To update a table, edit the CSV — no code changes needed.

The loaders are `get_random_coil`, `get_bmrb_stats`, `get_c_prime_rc`,
`get_csi_wishart`, and `get_panav_distns`.

## `random_coil.csv`

**Columns:** `residue, atom, value`

Random-coil chemical shifts for backbone atoms (CA, CB, N, H) per amino acid.
Empty `value` cells mean the atom does not exist for that residue (e.g. CB for
GLY, N/H for PRO) and load as `np.nan`.

**Source:** Wishart & Sykes (1994).

**Used by:** `get_random_coil()` — secondary-shift calculation (observed −
random coil) in `chemshift.py` and `reref/lacs.py` (LACS fitting).

## `bmrb_stats.csv`

**Columns:** `residue, atom, mean, std`

Full-database mean and standard deviation for backbone atoms (CA, CB, C, N, H)
per amino acid from the BMRB chemical-shift statistics.

**Source:** https://bmrb.io/ref_info/csstats.php?set=full&restype=aa (fetched 2026-04-17).

**Used by:** `get_bmrb_stats()` — flags shifts outside `mean ± n_std × std`
before LACS fitting in `reref/lacs.py`; flagged shifts are excluded from the
regression.

## `c_prime_rc.csv`

**Columns:** `residue, value`

Random-coil C′ (carbonyl carbon) chemical shifts per amino acid.

**Source:** Wishart et al. (1995).

**Used by:** `get_c_prime_rc()` — C′ secondary shift during LACS fitting in
`reref/lacs.py` (C′ uses its own random-coil reference, separate from CA/CB).

## `panav_distns.csv`

**Columns:** `Atom_name, SS, AA, mean, stdev`

Reference shift distributions per atom, secondary-structure class (C = coil,
H = helix, E = strand), and amino acid (Wang & Jardetzky 2002b). Used for
HA→SS assignment, Wang & Wishart (2005) offsets, and Wang et al. (2010) CONA
joint-probability scoring.

**Used by:** `get_panav_distns()` in `reref/panav.py`.

## `csi_wishart.csv`

**Columns:** `residue, atom, center, half_width`

Classic Chemical-Shift Index reference ranges (center ± half-width). Atoms
`HA` (Wishart, Sykes & Richards 1992), and `CA` / `CB` / `C` (Wishart & Sykes
1994, relative to TSP). Oxidized cysteine is stored as residue `CYSO`.

**Used by:** `get_csi_wishart()` — `makeshift.csi` / `ChemicalShifts.add_csi(method=...)`.

## `refdb/`

Optional Wishart [RefDB](https://refdb.wishartlab.com/) cache shipped with the
package (~45 MB):

| File | Contents |
|------|----------|
| `RefDB-{C,H,N}.db.txt` | SHIFTY backbone dumps |
| `refdb_backbone.csv` | Joined per-residue backbone table |
| `refdb_frags.npz` | Prebuilt fragment library |

Not loaded by default (PANAV uses `panav_distns.csv`). Available under
`makeshift/data/refdb/` for experiments that need RefDB.
