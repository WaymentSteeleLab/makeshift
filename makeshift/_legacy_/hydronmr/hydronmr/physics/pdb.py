"""
PDB parsing and residue-to-bead reduction.

Source: decompilations/hex-rays.txt
    pdbead_  19818  -- residue -> "AtoB" bead reduction, called by structure_pro__
    spipdb_  24550  -- (NOT used here: this is a SPIDER-format electron-
                       density-map reader, despite the name; it's the
                       branch of structure input used for EM-map modes,
                       not for PDB-protein mode 21/25.)

NOTE on translation approach
-----------------------------
The real `pdbead_` in HYDRONMR/AtoB encodes a large hard-coded table:
for each of the 20 standard amino acids (plus a few hetero groups), a
fixed number of "beads" (1-2 per residue) with template positions
relative to the backbone and tabulated hydrodynamic radii, calibrated
against atomic volumes. That table is hundreds of lines of DATA
statements in the original Fortran and is not recoverable from the
decompiled binary in any reasonable form (it shows up as opaque
floating-point constant blobs).

What's implemented here instead, to keep the pipeline runnable
end-to-end:

  * `parse_pdb_atoms`: a real, general PDB ATOM/HETATM record parser
    (this part *is* a faithful reimplementation -- the PDB fixed-column
    format is a public standard, independent of HYDRONMR's internals).

  * `pdbead_simple`: a *simplified* one-bead-per-residue reduction
    (bead at the residue's CA position, or atom centroid if no CA),
    with per-residue-type radii drawn from average amino-acid
    molecular volumes (Zamyatnin, 1972) via r = (3V/4*pi)^(1/3).
    This is dimensionally and physically sensible (each bead's volume
    approximates its residue's real volume) but is NOT the AtoB
    table, and will not numerically match GROUND_TRUTH bead-by-bead.
    It's flagged here so any later validation pass knows exactly what
    to replace if the real AtoB table can be recovered (e.g. from the
    HYDRONMR/AtoB Fortran source, which is published).
"""

from dataclasses import dataclass
import warnings
import numpy as np

# Average residue volumes in Angstrom^3 (Zamyatnin, 1972), used to
# derive a per-residue bead radius r = (3V / 4 pi)^(1/3).
RESIDUE_VOLUME_A3 = {
    "ALA": 88.6, "ARG": 173.4, "ASN": 114.1, "ASP": 111.1, "CYS": 108.5,
    "GLN": 143.8, "GLU": 138.4, "GLY": 60.1, "HIS": 153.2, "ILE": 166.7,
    "LEU": 166.7, "LYS": 168.6, "MET": 162.9, "PHE": 189.9, "PRO": 112.7,
    "SER": 89.0, "THR": 116.1, "TRP": 227.8, "TYR": 193.6, "VAL": 140.0,
}
DEFAULT_RESIDUE_VOLUME_A3 = 130.0  # fallback for unknown/hetero residues


@dataclass
class Atom:
    serial: int
    name: str
    resname: str
    chain: str
    resseq: int
    x: float
    y: float
    z: float
    element: str


def parse_pdb_atoms(path: str):
    """Parse ATOM/HETATM records from a PDB file (fixed-column format,
    columns per the PDB format spec). Returns a list of Atom."""
    atoms = []
    with open(path) as fh:
        for line in fh:
            rectype = line[0:6].strip()
            if rectype == "ENDMDL":
                break
            if rectype not in ("ATOM", "HETATM"):
                continue
            try:
                serial = int(line[6:11])
                name = line[12:16].strip()
                resname = line[17:20].strip()
                chain = line[21:22].strip()
                resseq = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                element = line[76:78].strip() or name[0]
            except ValueError:
                continue
            atoms.append(Atom(serial, name, resname, chain, resseq, x, y, z, element))

    if atoms:
        min_resseq = min(a.resseq for a in atoms)
        if min_resseq != 1:
            shift = 1 - min_resseq
            warnings.warn(
                f"PDB residue numbering started at {min_resseq}; "
                f"renumbering so it starts at 1 (shift = {shift:+d}).",
                RuntimeWarning,
            )
            for a in atoms:
                a.resseq += shift

    return atoms


def pdbead_uniform(atoms, aer_angstrom: float = 3.0):
    """Alternative reduction matching Fast-HYDRONMR's AER mode (see
    GROUND_TRUTH_DONT_OVERWRITE/*/hydronmr.dat 'AER, radius of the
    atomic elements'): one bead per heavy atom, all with the same
    radius `aer_angstrom`. Returns (positions, radii) in cm."""
    ANGSTROM_TO_CM = 1.0e-8
    positions = np.array([[a.x, a.y, a.z] for a in atoms if a.element != "H"])
    radii = np.full(len(positions), aer_angstrom)
    return positions * ANGSTROM_TO_CM, radii * ANGSTROM_TO_CM


def nh_bond_vectors(atoms):
    """Approximate per-residue backbone amide N-H bond unit vectors.

    Crystal structures (and Fast-HYDRONMR's bead model) generally lack
    explicit amide hydrogens. This estimates the N-H direction from the
    standard sp2 geometry at the backbone N: N is bonded to C(i-1) (the
    previous residue's carbonyl carbon), CA(i), and H, all ~120 deg
    apart in the peptide plane. So the N-H bond points roughly opposite
    the bisector of the C(i-1)-N-CA(i) angle:

        nh_hat = -normalize( normalize(N - Cprev) + normalize(N - CA) )

    Returns a dict keyed by (chain, resseq) -> unit vector (3,), for
    every residue that has N, CA, and a preceding residue's C atom
    (residue 1 of each chain has no C(i-1) and is skipped, matching
    tmp.t12's "190 out of 206" / similar undercounts).
    """
    # group atoms by (chain, resseq), preserving order
    residues = {}
    order = []
    for a in atoms:
        key = (a.chain, a.resseq)
        if key not in residues:
            residues[key] = {}
            order.append(key)
        residues[key][a.name] = np.array([a.x, a.y, a.z])

    vectors = {}
    prev_c = None
    for key in order:
        res = residues[key]
        n = res.get("N")
        ca = res.get("CA")
        c = res.get("C")
        if n is not None and ca is not None and prev_c is not None:
            v1 = n - prev_c
            v2 = n - ca
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            nh = -(v1 + v2)
            norm = np.linalg.norm(nh)
            if norm > 1e-8:
                vectors[key] = nh / norm
        prev_c = c if c is not None else prev_c
    return vectors


def pdbead_simple(atoms):
    """Simplified port of pdbead_ (hex-rays.txt:19818): reduce a list of
    Atom records to one hydrodynamic bead per residue.

    Each residue's bead is placed at its CA atom position (CA for
    protein residues; for residues without a CA, the centroid of all
    their atoms is used). The bead radius is derived from the
    residue's average molecular volume (RESIDUE_VOLUME_A3) via
    r = (3V / 4*pi)^(1/3), in Angstrom, then converted to cm for use
    with covol_/hydro_ (which expect cgs units, viscosity in poise).

    Returns (positions, radii) as numpy arrays, positions in cm.
    """
    residues = {}
    order = []
    for a in atoms:
        key = (a.chain, a.resseq, a.resname)
        if key not in residues:
            residues[key] = []
            order.append(key)
        residues[key].append(a)

    positions = []
    radii = []
    ANGSTROM_TO_CM = 1.0e-8

    for key in order:
        chain, resseq, resname = key
        res_atoms = residues[key]
        ca = next((a for a in res_atoms if a.name == "CA"), None)
        if ca is not None:
            pos = np.array([ca.x, ca.y, ca.z])
        else:
            coords = np.array([[a.x, a.y, a.z] for a in res_atoms])
            pos = coords.mean(axis=0)

        vol = RESIDUE_VOLUME_A3.get(resname, DEFAULT_RESIDUE_VOLUME_A3)
        r_angstrom = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)

        positions.append(pos * ANGSTROM_TO_CM)
        radii.append(r_angstrom * ANGSTROM_TO_CM)

    return np.array(positions), np.array(radii)
