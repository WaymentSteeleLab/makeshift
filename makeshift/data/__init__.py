"""Bundled reference tables (random-coil shifts, PANAV distributions, BMRB stats)."""

from .tables import (
    get_random_coil,
    get_panav_distns,
    get_bmrb_stats,
    get_c_prime_rc,
)

__all__ = [
    "get_random_coil",
    "get_panav_distns",
    "get_bmrb_stats",
    "get_c_prime_rc",
]
