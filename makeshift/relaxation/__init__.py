"""
CPMG relaxation-dispersion pipeline.

Turns a series of 2D ¹H-¹⁵N planes (a reference plane + CPMG planes at varying
νCPMG) into per-residue R₂,eff dispersion data, then classifies residues using
hydrodynamic predictions. Built on :mod:`makeshift.spectra` for the spectrum
and matching primitives, and :mod:`makeshift.hydronmr` for predicted R2.

Depends on scipy / tqdm / nmrglue (the optional ``relaxation`` extra), plus
matplotlib only if plotting is requested. Not imported by the top-level
package, so the core stays dependency-light.

    from makeshift.relaxation import run_protein
    results = run_protein("exp.yml", "out/")
"""

from .config import load_config, load_planes
from .lineshape import fit_peaks
from .r2eff import compute_r2eff
from .classify import classify_peaks, validate_sequence
from .cpmg import run_protein, save_results_csv

__all__ = [
    "run_protein", "save_results_csv",
    "load_config", "load_planes",
    "fit_peaks", "compute_r2eff",
    "classify_peaks", "validate_sequence",
]
