"""Unit handling when relaxation times are converted to rates (offline)."""
import numpy as np
import pandas as pd

from makeshift.relaxation.relax_profile import RelaxationProfile


def _df(val, err):
    return pd.DataFrame({"Seq_ID": [1, 2], "Atom_ID": ["N", "N"],
                         "Val": val, "Val_err": err})


def test_ms_time_error_converted_to_s_inverse():
    # T1 = 800 +/- 8 ms  ->  R1 = 1.25 +/- 0.0125 s^-1
    rate, err = RelaxationProfile._align_rate(_df([800.0, 400.0], [8.0, 8.0]),
                                              2, "ms", "T1")
    np.testing.assert_allclose(rate, [1.25, 2.5])
    np.testing.assert_allclose(err, [0.0125, 0.05])


def test_seconds_time_error_converted_to_s_inverse():
    rate, err = RelaxationProfile._align_rate(_df([0.8, 0.4], [0.008, 0.008]),
                                              2, "s", "T1")
    np.testing.assert_allclose(rate, [1.25, 2.5])
    np.testing.assert_allclose(err, [0.0125, 0.05])


def test_rate_error_passed_through():
    rate, err = RelaxationProfile._align_rate(_df([1.25, 2.5], [0.02, 0.03]),
                                              2, "s-1", "T1")
    np.testing.assert_allclose(rate, [1.25, 2.5])
    np.testing.assert_allclose(err, [0.02, 0.03])
