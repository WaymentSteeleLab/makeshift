"""
HYDRONMR engine: run state + Python API entry point.

Example
-------
    from makeshift.hydronmr import run
    result = run("in.pdb", csv_path="t1t2.csv")
    print(result.csv_path)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .physics.config import Config, DEFAULT_CONFIG_PATH
from .physics.structure import structure_pro
from .physics import tensors
from .physics.nmr import rotani, dipolesnmr
from .physics.pdb import nh_bond_vectors, parse_pdb_atoms
from .physics.output import write_t1t2_csv, per_residue_results


@dataclass
class HydronmrState:

    # ---- run-mode flags (set in MAIN__) ----------------------------------
    modehyd: int = 25      # 25 = "advanced" PDB/protein NMR mode

    # ---- diffusion tensors / hydrodynamics --------------------------------
    dt_trans: Optional[np.ndarray] = None   # 3x3 translational diffusion tensor
    dt_rot: Optional[np.ndarray] = None     # 3x3 rotational diffusion tensor
    center_of_diffusion: Optional[np.ndarray] = None
    center_of_mass: Optional[np.ndarray] = None

    # ---- bead model (populated by physics/structure.py) --------------------
    bead_positions: Optional[np.ndarray] = None   # (N,3) cm
    bead_radii: Optional[np.ndarray] = None       # (N,) cm
    friction_translation: Optional[np.ndarray] = None  # 3x3
    friction_rotation: Optional[np.ndarray] = None     # 3x3
    friction_coupling: Optional[np.ndarray] = None     # 3x3
    friction_full: Optional[np.ndarray] = None         # 3N x 3N
    diffusion_66: Optional[np.ndarray] = None          # 6x6

    # ---- physical parameters -------------------------------------------------
    temperature: float = 293.15   # K
    viscosity: float = 0.01       # poise

    # ---- I/O ------------------------------------------------------------------
    log_lines: List[str] = field(default_factory=list)
    aer: float = 3.0             # uniform atomic-element bead radius, Angstrom
    pdb_path: str = ""

    def log(self, *args):
        line = " ".join(str(a) for a in args)
        self.log_lines.append(line)
        print(line)


@dataclass
class Result:
    state: HydronmrState
    per_residue: dict       # (chain, resseq) -> (T1, T2, T1/T2, NOE)
    csv_path: Optional[Path] = None


def run(
    pdb_path: Union[str, Path],
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
    csv_path: Optional[Union[str, Path]] = None,
) -> Result:
    """Run the mode-25 pipeline (PDB -> diffusion tensor -> per-residue
    T1/T2/NOE) for `pdb_path`, using parameters from `config_path`
    (default: `python_port/config.yml`).

    If `csv_path` is given, also write per-residue results there
    (columns: resseq, T1, T2, T1_over_T2, NOE).
    """
    cfg = Config.from_yaml(config_path)

    g = HydronmrState()
    g.modehyd = 25
    g.pdb_path = str(pdb_path)
    g.aer = cfg.aer_angstrom
    g.temperature = cfg.temperature_k
    g.viscosity = cfg.viscosity_poise

    structure_pro(g)

    ct, cr, cc, c, cod = tensors.hydro(
        g.bead_positions, g.bead_radii, g.viscosity,
        center=g.center_of_mass, ind=cfg.ind,
    )
    g.friction_translation = ct
    g.friction_rotation = cr
    g.friction_coupling = cc
    g.friction_full = c
    g.center_of_diffusion = cod

    tensors.generdif66_state(g)
    rotani(g)
    dipolesnmr(
        g,
        b0_tesla=cfg.fields_tesla[0],
        gamma_x=cfg.gamma_x_e7 * 1.0e7,
        r_nh_angstrom=cfg.r_nh_angstrom,
        csa_ppm=cfg.csa_ppm,
    )

    atoms = parse_pdb_atoms(g.pdb_path)
    nh_vectors = nh_bond_vectors(atoms)

    per_residue = per_residue_results(g, nh_vectors)

    written = None
    if csv_path is not None:
        written = write_t1t2_csv(g, nh_vectors, csv_path)

    return Result(state=g, per_residue=per_residue, csv_path=written)
