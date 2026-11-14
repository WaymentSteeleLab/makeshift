"""
ChemEx: fit CPMG relaxation dispersion (exchange rates, populations, chemical
shift differences) by running the ChemEx program on makeshift's peak fits.
"""

from . import utils
from .engine import ChemEx, chemex_command
from .utils import save_chemex

__all__ = ["ChemEx", "chemex_command", "save_chemex", "utils"]
