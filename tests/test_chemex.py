"""Round-trip test for the ChemEx wrapper: simulate a 2-state CPMG dataset with
ChemEx itself, feed it back through ChemEx.from_fit_results as if it came from
fit_peaks, and check the known kex / pB / dw are recovered."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("chemex")

from makeshift.chemex import ChemEx, chemex_command, utils

TIME_T2 = 0.04
NCYCS = [0, 1, 2, 4, 8, 10, 16, 20, 30, 40]
TRUE = {"PB": 0.05, "KEX_AB": 800.0, "DW_AB": {"G2N": 2.0, "A3N": 0.5}}
SHIFTS = {"G2N": 115.0, "A3N": 122.0}
EXPERIMENT = {"name": "cpmg_15n_tr", "time_t2": TIME_T2, "carrier": 118.0, "pw90": 35e-6}
CONDITIONS = {"h_larmor_frq": 800.0, "temperature": 25.0}


def _simulate(tmp_path):
    """Noise-free intensities for G2/A3 from ChemEx's own simulator."""
    flat = pd.DataFrame({"ncyc": NCYCS, "intensity": 1e5, "error": 1e3})
    data = tmp_path / "data"
    utils.write_profiles({"G2N-H": flat, "A3N-H": flat}, data)
    utils.write_experiment_toml(tmp_path / "exp.toml", EXPERIMENT, CONDITIONS,
                                {"G2N-H": "G2N-H.out", "A3N-H": "A3N-H.out"}, "data")
    utils.write_parameters_toml(tmp_path / "p.toml", {
        "GLOBAL": {"PB": TRUE["PB"], "KEX_AB": TRUE["KEX_AB"], "R2_A": 10.0},
        "CS_A": SHIFTS, "DW_AB": TRUE["DW_AB"]})
    cmd = [*chemex_command(), "simulate", "-e", "exp.toml", "-p", "p.toml",
           "-o", "Sim", "--plot", "nothing"]
    import subprocess
    subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True)
    return utils.parse_dat(tmp_path / "Sim" / "Data" / "exp.dat")


def test_roundtrip_recovers_exchange(tmp_path):
    sim = _simulate(tmp_path)
    rng = np.random.default_rng(1)
    plane = {n: i for i, n in enumerate(NCYCS)}
    rows = []
    for ref_index, (_, g) in enumerate(sim.groupby("profile", sort=False)):
        for n, c in zip(g["ncyc"], g["calc"]):
            rows.append(dict(ref_index=ref_index, plane=plane[n],
                             vol=c + rng.normal(0, 1500), vol_err=1500.0))
    fit_results = pd.DataFrame(rows)
    vcpmg = [n / TIME_T2 for n in NCYCS[1:]]
    ref_df = pd.DataFrame({"ref_index": [0, 1], "assn_label": ["G2", "A3"],
                           "N_ppm": [115.0, 122.0]})

    cx = ChemEx.from_fit_results(fit_results, vcpmg, TIME_T2, ref_df,
                                 pw90=35e-6, temperature=25.0,
                                 h_larmor_frq=800.0, carrier=118.0)
    cx.run(output_dir=tmp_path / "fit")

    g = cx.global_parameters
    assert abs(g.loc["PB", "value"] - TRUE["PB"]) < 3 * g.loc["PB", "error"]
    assert abs(g.loc["KEX_AB", "value"] - TRUE["KEX_AB"]) < 3 * g.loc["KEX_AB", "error"]

    r = cx.residue_parameters
    assert list(r.index) == ["G2N", "A3N"]
    assert list(r["ref_index"]) == [0, 1]
    for spin, dw in TRUE["DW_AB"].items():
        assert abs(abs(r.loc[spin, "DW_AB"]) - dw) < 3 * r.loc[spin, "DW_AB_err"]

    prof = cx.fit_profiles
    assert set(prof["profile"]) == {"G2N-H", "A3N-H"}
    assert np.allclose(prof.loc[prof["ncyc"] > 0, "vcpmg"].unique(), vcpmg)
    assert cx.statistics["reduced-chi-square"] < 3


def test_parameter_overrides_and_method():
    prof = {"G2N-H": pd.DataFrame({"ncyc": NCYCS, "intensity": 1.0, "error": 0.1})}
    cx = ChemEx(prof, EXPERIMENT, CONDITIONS, shifts={"G2N": 115.0},
                parameters={"GLOBAL": {"KEX_AB": 2000.0}, "R2_A": 12.0},
                method=["PB", "KEX_AB"])
    p = cx.build_parameters()
    assert p["GLOBAL"]["KEX_AB"] == 2000.0 and p["GLOBAL"]["PB"] == [0.05, 0.0, 0.5]
    assert p["CS_A"] == {"G2N": 115.0} and p["R2_A"] == {"G2N": 12.0}
    assert cx.build_method() == {"STEP1": {"ROLES": [{"FIT": ["PB", "KEX_AB"]}]}}


def test_failure_reports_chemex_output():
    prof = {"G2N-H": pd.DataFrame({"ncyc": NCYCS, "intensity": 1.0, "error": 0.1})}
    bad = {k: v for k, v in EXPERIMENT.items() if k != "pw90"}
    with pytest.raises(RuntimeError, match="pw90"):
        ChemEx(prof, bad, CONDITIONS, shifts={"G2N": 115.0}).run()
