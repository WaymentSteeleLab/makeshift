"""
R1/R2/NOE and CPMG relaxation-dispersion pipeline
"""

from .config import load_config, load_planes
from .lineshape import fit_peaks
from .r2eff import compute_r2eff
from .classify import classify_peaks, validate_sequence
from .cpmg import CPMGExperiment, run_protein, save_results_csv
from .relax_profile import RelaxationProfile

__all__ = [
    "CPMGExperiment", "run_protein", "save_results_csv",
    "RelaxationProfile",
    "load_config", "load_planes",
    "fit_peaks", "compute_r2eff",
    "classify_peaks", "validate_sequence",
]