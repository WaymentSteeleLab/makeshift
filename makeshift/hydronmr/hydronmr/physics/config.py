"""
Run-configuration loading.

Replaces per-run parsing of `hydronmr.dat` with a single shared
`config.yml` (see `python_port/config.yml` for the default). The only
thing that should vary between runs is the input PDB structure, which
is passed separately to `hydronmr.run(pdb_path, ...)`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yml"


@dataclass
class Config:
    temperature_k: float = 293.0
    viscosity_poise: float = 0.01
    aer_angstrom: float = 3.0
    ind: int = 1
    nsig: int = -1

    iflag_dipoles: int = 1
    gamma_x_e7: float = -2.7126
    r_nh_angstrom: float = 1.02
    csa_ppm: float = -172.0
    fields_tesla: List[float] = field(default_factory=lambda: [11.74])

    @classmethod
    def from_yaml(cls, path=DEFAULT_CONFIG_PATH) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)
