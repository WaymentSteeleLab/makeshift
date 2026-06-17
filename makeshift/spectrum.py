# Re-exports for backwards compatibility — ms.spectrum.X still works.
# Implementations now live in io.py, peaks.py, matching.py, plotting.py.

from .io import Spectrum, read_ucsf, estimate_background, _annotate_ppm
from .peaks import pick_peaks, load_peaklist, _AA_3TO1
from .matching import map_peaklists, match_peaks_hungarian_, _find_offset, _detect_conflicts, _apply_how
from .plotting import plot_spectrum, plot_peaklist
