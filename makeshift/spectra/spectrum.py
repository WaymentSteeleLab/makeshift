"""
2D NMR spectra: reading Sparky ``.ucsf`` files and spectrum-level operations.

Depends on ``nmrglue`` will need to import if needed
"""

import numpy as np
import pandas as pd
import nmrglue as ng

def estimate_background(data, n=1000, seed=42):
    """Noise-floor estimate: median absolute intensity over ``n`` random
    points of ``data``. Deterministic for a given ``seed``."""
    rng = np.random.default_rng(seed)
    idx = tuple(rng.integers(0, dim, size=n) for dim in data.shape)
    return float(np.median(np.abs(data[idx])))


def _annotate_ppm(peaks_table, uc):
    """Turn an nmrglue peak table into a tidy DataFrame with ppm columns."""
    df = pd.DataFrame(peaks_table)
    df["N_ppm"] = df["Y_AXIS"].apply(lambda r: uc[0].ppm(r))
    df["H_ppm"] = df["X_AXIS"].apply(lambda r: uc[1].ppm(r))
    df.rename(columns={"Y_AXIS": "N_axis", "X_AXIS": "H_axis",
                       "Y_LW": "N_lw", "X_LW": "H_lw",
                       "VOL": "est_vol", "cID": "cid"}, inplace=True)
    return df


class Spectrum:
    """
    A 2D spectrum loaded from a Sparky ``.ucsf`` file: the data array plus its
    per-axis unit-conversion objects (for ppm <-> point conversions).

    Attributes
    ----------
    data : np.ndarray
        The intensity array (axis 0 = indirect/N, axis 1 = direct/H).
    uc : list
        One nmrglue unit-conversion object per axis.
    """

    def __init__(self, data, uc):
        self.data = data
        self.uc = uc

    @classmethod
    def from_ucsf(cls, path):
        """Read a Sparky ``.ucsf`` file into a Spectrum."""
        dic, data = ng.sparky.read(str(path))
        data = data.astype(float)
        uc = [ng.sparky.make_uc(dic, data, dim=i) for i in range(data.ndim)]
        return cls(data, uc)

    def estimate_background(self, n=1000, seed=42):
        """
        Noise-floor estimate: median absolute intensity over ``n`` randomly
        sampled points. Deterministic for a given ``seed``.
        """
        return estimate_background(self.data, n=n, seed=seed)

    def pick_peaks(self, baseline=10, algorithm="downward", est_params=True,
                   h_ppm_min=6.0, h_ppm_max=11.0):
        """
        Pick peaks in the amide region of this 2D ¹H-¹⁵N spectrum.

        Peaks above ``baseline`` × the noise floor (estimated within the ¹H
        window) are picked with nmrglue.

        Parameters
        ----------
        baseline : float
            Threshold multiple of the noise floor.
        algorithm : str
            nmrglue peak-picking algorithm ('downward' or 'connected').
        est_params : bool
            Have nmrglue estimate linewidths/volumes.
        h_ppm_min, h_ppm_max : float
            ¹H window to search (ppm).

        Returns
        -------
        DataFrame
            One row per picked peak with columns N_axis, H_axis, N_lw, H_lw,
            est_vol, cid, N_ppm, H_ppm. ``est_vol`` is the picker's estimate —
            treat as approximate (the lineshape fit refines it).
        """
        data = np.asarray(self.data, dtype=float)
        ppm_h = self.uc[1].ppm_scale()
        cols = np.where((ppm_h >= h_ppm_min) & (ppm_h <= h_ppm_max))[0]
        col_start, col_stop = int(cols[0]), int(cols[-1]) + 1
        data_slice = data[:, col_start:col_stop]

        threshold = baseline * estimate_background(data_slice)
        peaks_table = ng.analysis.peakpick.pick(
            data_slice, pthres=threshold, nthres=None,
            algorithm=algorithm, est_params=est_params,
            diag=False, edge=(0, col_start), table=True,
        )
        return _annotate_ppm(peaks_table, self.uc)

    def ppm(self, axis, point):
        """Convert a point index on ``axis`` to ppm."""
        return self.uc[axis].ppm(point)

    def __repr__(self):
        shape = "x".join(map(str, self.data.shape))
        return f"Spectrum(shape={shape}, dims={len(self.uc)})"
