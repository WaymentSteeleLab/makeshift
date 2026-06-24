import numpy as np
import pandas as pd
import nmrglue as ng
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

@dataclass
class Spectrum:
    """A 2D spectrum loaded from a Sparky .ucsf file: the data array plus
    its per-axis unit-conversion objects (for ppm <-> point conversions)."""
    data: np.ndarray
    uc: list


def read_ucsf(fpath):
    """Read a Sparky .ucsf file. Returns a Spectrum (data + uc)."""
    dic, data = ng.sparky.read(fpath)
    data = data.astype(float)
    uc = [ng.sparky.make_uc(dic, data, dim=i) for i in range(data.ndim)]
    return Spectrum(data, uc)


def estimate_background(data, seed=42):
    """Randomly sample 1000 points and take median absolute value as noise floor."""
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, data.shape[0], size=1000)
    cols = rng.integers(0, data.shape[1], size=1000)
    return float(np.median(np.abs(data[rows, cols])))


def _annotate_ppm(peaks_table, uc):
    df = pd.DataFrame(peaks_table)
    df["N_ppm"] = df["Y_AXIS"].apply(lambda r: uc[0].ppm(r))
    df["H_ppm"] = df["X_AXIS"].apply(lambda r: uc[1].ppm(r))
    df.rename(columns={"Y_AXIS": "N_axis", "X_AXIS": "H_axis",
                       "Y_LW": "N_lw", "X_LW": "H_lw",
                       "VOL": "est_vol", "cID": "cid"}, inplace=True)
    return df
