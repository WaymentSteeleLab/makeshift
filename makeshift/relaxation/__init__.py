"""
R1/R2/NOE and CPMG relaxation-dispersion pipeline
"""

from .config import load_config, load_planes
from .lineshape import fit_peaks
from .r2eff import compute_r2eff
from ..chemex.utils import save_chemex
from .classify import classify_peaks, validate_sequence
from .cpmg import CPMGExperiment, run_protein, save_results_csv
from .relax_profile import RelaxationProfile
from .model_free import FittingError

__all__ = [
    "CPMGExperiment", "run_protein", "save_results_csv",
    "RelaxationProfile", "FittingError",
    "load_config", "load_planes",
    "fit_peaks", "compute_r2eff", "save_chemex",
    "classify_peaks", "validate_sequence",
]