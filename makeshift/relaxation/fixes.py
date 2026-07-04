"""
Known BMRB deposit fixes for backbone relaxation data.

A handful of legacy entries declare incorrect or missing units, swap the
value and error columns, or otherwise need a one-off correction to line
up with the rest of the archive. These were identified empirically (by
checking against independently known relaxation rates) while curating
RelaxDB — see Wayment-Steele, El Nesr et al., "Learning millisecond
protein dynamics from what is missing in NMR spectra".

Tables are keyed by (entry_id, kind), where kind is "T1" or "T2" (matching
NMRStarEntry.relaxation's kind names); entries with no kind dependence are
keyed by entry_id alone.
"""

# Forces the units string when the deposited tag is wrong or absent ('.').
_UNIT_OVERRIDES = {
    5154: "s",
    7056: "s",
    7088: "s",
    15930: "s",
    16737: "ms",
    17701: "ms",
    17266: "s-1",
    18087: "s-1",
    18758: "s-1",
    50734: "ms",
    50745: "ms",
    51478: "s-1",
    51126: "s-1",
    (51306, "T1"): "s",
    (51306, "T2"): "s-1",
}

# Value and error columns are swapped in the deposit.
_VALUE_ERR_SWAPPED = {
    (26788, "T2"),
    (17266, "T2"),
}

# Error needs an extra plain reciprocal after the normal time->rate
# conversion (flipped error, confirmed with the depositing authors).
_ERR_RECIPROCAL = {
    (27011, "T2"),
}

# The "error" column actually holds a second value column (e.g. R1/R2
# instead of an uncertainty on T1/T2), so it isn't usable as an error.
_ERR_INVALID = {
    15486,
}


def _as_int(entry_id):
    try:
        return int(entry_id)
    except (TypeError, ValueError):
        return entry_id


def _lookup(table, entry_id, kind):
    entry_id = _as_int(entry_id)
    return (entry_id, kind) in table or entry_id in table


def unit_override(entry_id, kind):
    """Forced units string for (entry_id, kind), or None if no override applies."""
    entry_id = _as_int(entry_id)
    if (entry_id, kind) in _UNIT_OVERRIDES:
        return _UNIT_OVERRIDES[(entry_id, kind)]
    return _UNIT_OVERRIDES.get(entry_id)


def value_err_swapped(entry_id, kind):
    return _lookup(_VALUE_ERR_SWAPPED, entry_id, kind)


def err_reciprocal(entry_id, kind):
    return _lookup(_ERR_RECIPROCAL, entry_id, kind)


def err_invalid(entry_id, kind):
    return _lookup(_ERR_INVALID, entry_id, kind)
