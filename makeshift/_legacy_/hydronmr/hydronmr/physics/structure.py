"""
Structure-input routines: build the bead/atom model from the chosen
input representation.

Source: decompilations/hex-rays.txt
    structure_hyd__  line 24136  (mode 11: explicit bead-model file)
    structure_mic__  line 24245  (mode 31: micelle/aggregate)
    structure_pro__  line 25437  (mode 21/25: PDB protein structure -> AtoB beads)
    structure_sub__  line 25479  (mode 41: substructure assembly)
    structure_pix__  line 26006  (mode 34: pixel/voxel map)
"""

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..engine import HydronmrState
from .pdb import parse_pdb_atoms, pdbead_uniform


def structure_hyd(g: "HydronmrState"):
    raise NotImplementedError


def structure_mic(g: "HydronmrState"):
    raise NotImplementedError


def structure_pro(g: "HydronmrState"):
    """Port of structure_pro__ (hex-rays.txt:25437).

    For modehyd_ in (21, 25) -- the case relevant to the default
    MAIN__ configuration (modehyd_ == 25) -- this reads the PDB file
    named at g.pdb_path and reduces it to one bead per heavy atom,
    all with radius g.aer Angstrom (Fast-HYDRONMR's "AER, radius of
    the atomic elements" mode -- see routines/pdb.pdbead_uniform and
    GROUND_TRUTH_DONT_OVERWRITE/*/hydronmr.dat). Populates
    g.bead_positions / g.bead_radii / g.center_of_mass for the rest
    of the pipeline (tensors.hydro, tensors.generdif66_state, etc).
    """
    atoms = parse_pdb_atoms(g.pdb_path)
    if not atoms:
        raise ValueError(f"structure_pro_: no ATOM/HETATM records found in {g.pdb_path}")

    positions, radii = pdbead_uniform(atoms, g.aer)
    g.bead_positions = positions
    g.bead_radii = radii
    g.center_of_mass = positions.mean(axis=0)
    g.center_of_diffusion = g.center_of_mass
    g.log(f"structure_pro_: read {len(atoms)} atoms, {len(radii)} beads "
          f"(AER={g.aer} Angs) from {g.pdb_path}")


def structure_sub(g: "HydronmrState"):
    raise NotImplementedError


def structure_pix(g: "HydronmrState"):
    raise NotImplementedError
