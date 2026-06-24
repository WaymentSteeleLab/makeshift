# Reference data

All files are loaded at import time by `makeshift/utils/tables.py` and cached as
module-level dicts. To update a table, edit the CSV — no code changes needed.

---

## `random_coil.csv`

**Columns:** `residue, atom, value`

Random coil chemical shifts for backbone atoms (CA, CB, N, H) per amino acid.
Empty `value` cells indicate the atom does not exist for that residue (e.g. CB
for GLY, N/H for PRO) and are loaded as `np.nan`.

**Source:** Wishart & Sykes (1994), updated values from jojo-use branch.

**Used by:** `get_random_coil()` → `chemshift_utils.get_secondary_shift()` — secondary
shift calculation (observed − random coil) for CSI and LACS fitting.

---

## `bmrb_stats.csv`

**Columns:** `residue, atom, mean, std`

Full-database mean and standard deviation for backbone atoms (CA, CB, C, N, H)
per amino acid, drawn from the BMRB chemical shift statistics.

**Source:** https://bmrb.io/ref_info/csstats.php?set=full&restype=aa (fetched 2026-04-17).

**Used by:** `get_bmrb_stats()` → `ReRef._is_outlier()` — flags observed shifts
that fall outside `mean ± n_std × std` before LACS fitting. Outlier rows receive
`reref_mask=False` and are excluded from regression.

---

## `c_prime_rc.csv`

**Columns:** `residue, value`

Random coil C' (carbonyl carbon) chemical shifts per amino acid.

**Source:** Wishart et al. (1995).

**Used by:** `get_c_prime_rc()` → `ReRef._c_prime_secondary_shift()` — secondary
shift for C' atoms during LACS fitting (C' is handled separately from CA/CB
because it uses its own random coil reference).

---

## `panav_distns.csv`

**Columns:** `Atom_name, SS, AA, mean, stdev`

Reference chemical shift distributions per atom, secondary structure class
(C = coil, H = helix, E = strand), and amino acid. Used to assign the most
probable secondary structure to each residue from its HA shift, which drives
the PANAV offset calculation.

**Source:** Wang & Wishart (2005); distributed as `wang_jardetsky_distns.csv`
in the original jojo-use scripts.

**Used by:** `get_panav_distns()` → `ReRef._panav_ss_probs()` and
`ReRef._panav_get_offset()` — probabilistic secondary structure assignment
and per-atom offset estimation during PANAV re-referencing.
