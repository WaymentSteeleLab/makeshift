"""Round-trip test for the ChemEx wrapper: simulate a 2-state CPMG dataset with
ChemEx itself, feed it back through ChemEx.from_fit_results as if it came from
fit_peaks, and check the known kex / pB / dw are recovered."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("chemex")

from makeshift.chemex import ChemEx, chemex_command, exchange_start, utils

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

    settings = {"experiment": "cpmg_15n_tr", "h_larmor_frq": 800.0,
                "carrier": 118.0, "temperature": 25.0, "pw90": 35e-6}
    config = {"time_T2": TIME_T2, "planes": [{"vcpmg": v} for v in vcpmg[::-1]],
              "chemex": settings}          # plane order in the yaml doesn't matter
    cx = ChemEx.from_fit_results(fit_results, ref_df, config)
    import os
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        cx.run(output_dir="fit")        # relative output_dir must work
    finally:
        os.chdir(cwd)

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
                parameters={"R2_A": 12.0}, method=["PB", "KEX_AB"])
    p = cx.build_parameters()
    assert p["GLOBAL"] == {"PB": [0.05, 0.0, 0.2], "KEX_AB": [500.0, 10.0, 10000.0]}
    p = cx.build_parameters(exchange=exchange_start(kex=2000, kex_bounds=None,
                                                    pb=0.1, pb_bounds=(0.01, 0.3)))
    assert p["GLOBAL"] == {"PB": [0.1, 0.01, 0.3], "KEX_AB": 2000.0}
    assert p["CS_A"] == {"G2N": [115.0, 114.5, 115.5]} and p["R2_A"] == {"G2N": 12.0}
    assert cx.build_parameters(cs_window=None)["CS_A"] == {"G2N": 115.0}
    assert ChemEx(prof, EXPERIMENT, CONDITIONS).build_method() == {
        "STEP1": {"ROLES": [{"FIT": ["PB", "KEX_AB", "DW_AB"]}]}}
    assert cx.build_method() == {"STEP1": {"ROLES": [{"FIT": ["PB", "KEX_AB"]}]}}


def test_failure_reports_chemex_output():
    prof = {"G2N-H": pd.DataFrame({"ncyc": NCYCS, "intensity": 1.0, "error": 0.1})}
    bad = {k: v for k, v in EXPERIMENT.items() if k != "pw90"}
    with pytest.raises(RuntimeError, match="pw90"):
        ChemEx(prof, bad, CONDITIONS, shifts={"G2N": 115.0}).run()


CONFIG_BASE = """\
time_T2: 0.04
data_dir: /nonexistent
reference: ref.ucsf
planes:
  - {file: p50.ucsf, vcpmg: 50}
"""


def _fake_cpmg(chemex_section):
    """A run() CPMGExperiment stand-in: config + fit results + a header."""
    from types import SimpleNamespace
    uc = [SimpleNamespace(_obs=60.8, _car=60.8 * 117.5),   # 15N axis
          SimpleNamespace(_obs=600.1, _car=600.1 * 4.7)]   # 1H axis
    fit_results = pd.DataFrame({"ref_index": [0, 0], "plane": [0, 1],
                                "vol": [1.0, 0.8], "vol_err": [0.01, 0.01]})
    ref_df = pd.DataFrame({"ref_index": [0], "assn_label": ["G2"], "N_ppm": [115.0]})
    config = {"time_T2": TIME_T2, "planes": [{"vcpmg": 25.0}]}
    if chemex_section is not None:
        config["chemex"] = chemex_section
    return SimpleNamespace(config=config, fit_results=fit_results, ref_df=ref_df,
                           vcpmg_values=[25.0], time_T2=TIME_T2,
                           ref_spectrum=SimpleNamespace(uc=uc))


def test_yaml_chemex_section_is_validated(tmp_path):
    from makeshift.relaxation.config import load_config
    path = tmp_path / "cpmg.yaml"

    path.write_text(CONFIG_BASE + "chemex:\n  temperature: 25\n  pw90: 3.5e-5\n"
                    "  parameters: {R2_A: 12}\n")
    s = load_config(path)["chemex"]
    assert s["experiment"] == "cpmg_15n_ip" and s["model"] == "2st"
    assert s["temperature"] == 25.0 and isinstance(s["temperature"], float)
    assert s["parameters"] == {"R2_A": 12}

    path.write_text(CONFIG_BASE + "chemex:\n  temperature: 25\n  pw90: 3.5e-5\n"
                    "  parameters: {GLOBAL: {KEX_AB: 1500}}\n")
    with pytest.raises(ValueError, match="kex"):          # kex/pB belong to run()
        load_config(path)

    path.write_text(CONFIG_BASE + "chemex:\n  temperature: 25\n  pw_90: 3.5e-5\n")
    with pytest.raises(ValueError, match="pw_90"):        # typo -> unknown key
        load_config(path)

    path.write_text(CONFIG_BASE + "chemex:\n  temperature: 25\n")
    with pytest.raises(ValueError, match="pw90"):         # required key missing
        load_config(path)

    path.write_text(CONFIG_BASE)                          # section is optional
    assert "chemex" not in load_config(path)


def test_from_cpmg_reads_yaml_settings_and_header():
    cx = ChemEx.from_cpmg(_fake_cpmg({"temperature": 25.0, "pw90": 35e-6,
                                      "experiment_settings": {"antitrosy": True},
                                      "method": ["R2_A"]}))
    assert cx.conditions == {"h_larmor_frq": 600.1, "temperature": 25.0}
    assert cx.experiment["carrier"] == pytest.approx(117.5)
    assert cx.experiment["time_t2"] == TIME_T2 and cx.experiment["antitrosy"] is True
    assert cx.shifts == {"G2N": 115.0} and cx.method == ["R2_A"]

    # explicit YAML values win over the header; keyword overrides win over YAML
    cx = ChemEx.from_cpmg(_fake_cpmg({"temperature": 25.0, "pw90": 35e-6,
                                      "h_larmor_frq": 800.0, "carrier": 119.0}),
                          model="1st")
    assert cx.conditions["h_larmor_frq"] == 800.0 and cx.experiment["carrier"] == 119.0
    assert cx.model == "1st"

    with pytest.raises(ValueError, match="chemex"):
        ChemEx.from_cpmg(_fake_cpmg(None))


def _simulated_chemex(tmp_path):
    """ChemEx object holding the noisy simulated G2/A3 profiles."""
    sim = _simulate(tmp_path)
    rng = np.random.default_rng(1)
    profiles = {}
    for name, g in sim.groupby("profile", sort=False):
        profiles[name] = pd.DataFrame({
            "ncyc": g["ncyc"].to_numpy(),
            "intensity": g["calc"].to_numpy() + rng.normal(0, 1500, len(g)),
            "error": 1500.0})
    return ChemEx(profiles, EXPERIMENT, CONDITIONS, shifts=SHIFTS)


def test_select_residues():
    prof = {n: pd.DataFrame({"ncyc": NCYCS, "intensity": 1.0, "error": 0.1})
            for n in ("G2N-H", "A3N-H", "K10N-H")}
    cx = ChemEx(prof, EXPERIMENT, CONDITIONS)
    assert cx.select() == ["G2N-H", "A3N-H", "K10N-H"]
    assert cx.select(["K10", "g2"]) == ["G2N-H", "K10N-H"]       # profile order
    assert cx.select([3, "10"]) == ["A3N-H", "K10N-H"]           # residue numbers
    assert cx.select("A3N") == ["A3N-H"] and cx.select(["A3N-H"]) == ["A3N-H"]
    with pytest.raises(ValueError, match="Y99"):
        cx.select(["G2", "Y99"])
    # per-spin overrides only for the selected spins
    cx.parameters = {"DW_AB": {"G2N": 3.0, "K10N": 0.2}}
    p = cx.build_parameters(cx.select(["G2"]))
    assert p["DW_AB"] == {"G2N": 3.0}


def test_global_fit_of_selected_residues(tmp_path):
    cx = _simulated_chemex(tmp_path)
    cx.run(residues=["G2"], fit="global", output_dir=tmp_path / "g")
    assert cx.results["spins"] == ["G2N"]
    assert list(cx.residue_parameters.index) == ["G2N"]
    assert set(cx.fit_profiles["profile"]) == {"G2N-H"}
    assert {"KEX_AB", "PB"} <= set(cx.global_parameters.index)


def test_individual_fits(tmp_path):
    cx = _simulated_chemex(tmp_path)
    cx.run(fit="individual", output_dir=tmp_path / "ind")
    assert cx.global_parameters is None
    r = cx.residue_parameters
    assert list(r.index) == ["G2N", "A3N"]
    for col in ("KEX_AB", "KEX_AB_err", "PB", "PB_err", "DW_AB", "R2_A"):
        assert col in r.columns and r[col].notna().all(), col
    # G2 has the big dw, so its own fit should still find the true exchange
    assert abs(r.loc["G2N", "KEX_AB"] - TRUE["KEX_AB"]) < 3 * r.loc["G2N", "KEX_AB_err"]
    assert list(cx.statistics.index) == ["G2N", "A3N"]
    assert set(cx.fit_profiles["profile"]) == {"G2N-H", "A3N-H"}
    assert (tmp_path / "ind" / "G2N" / "chemex").is_dir()
    with pytest.raises(ValueError, match="global"):
        cx.run(fit="pooled")


def test_from_fit_results_with_yaml_path_and_acqus(tmp_path):
    """Only fit_results, the peak table and the config path are needed."""
    from test_acqus import _acqus_text
    (tmp_path / "acqus").write_text(_acqus_text())
    cfg = tmp_path / "cpmg.yaml"
    cfg.write_text(f"data_dir: {tmp_path}\nacqus: acqus\ntime_T2: 0.04\nreference: ref.ucsf\n"
                   "planes:\n  - {file: a.ucsf, vcpmg: 50}\n  - {file: b.ucsf, vcpmg: 25}\n")
    fit_results = pd.DataFrame({"ref_index": 0, "plane": [0, 1, 2],
                                "vol": [1.0, 0.8, 0.9], "vol_err": 0.01})
    ref_df = pd.DataFrame({"ref_index": [0], "assn_label": ["G2"], "N_ppm": [115.0]})
    cx = ChemEx.from_fit_results(fit_results, ref_df, cfg)
    assert cx.experiment["name"] == "cpmg_15n_ip"
    assert cx.experiment["time_t2"] == pytest.approx(0.04)          # from the yaml
    assert cx.experiment["time_equil"] == pytest.approx(0.005)      # d20
    assert cx.conditions["temperature"] == pytest.approx(25.0)
    # planes sorted by vcpmg: plane 1 = 25 Hz (ncyc 1), plane 2 = 50 Hz (ncyc 2)
    assert list(cx.profiles["G2N-H"]["ncyc"]) == [0.0, 1.0, 2.0]
    assert ChemEx.from_fit_results(fit_results, ref_df, cfg, model="1st").model == "1st"


def test_exchange_start_validation():
    assert exchange_start() == {"PB": [0.05, 0.0, 0.2], "KEX_AB": [500.0, 10.0, 10000.0]}
    assert exchange_start(kex_bounds=(None, 5000))["KEX_AB"] == [500.0, -float("inf"), 5000.0]
    with pytest.raises(ValueError, match="outside"):
        exchange_start(pb=0.3)                        # above the default 0.2 cap
    with pytest.raises(ValueError, match="within"):
        exchange_start(pb_bounds=(0, 1.5))
    with pytest.raises(ValueError, match="min < max"):
        exchange_start(kex_bounds=(1000, 100))
    prof = {"G2N-H": pd.DataFrame({"ncyc": NCYCS, "intensity": 1.0, "error": 0.1})}
    with pytest.raises(ValueError, match="run"):
        ChemEx(prof, EXPERIMENT, CONDITIONS, parameters={"GLOBAL": {"pb": 0.1}})


def test_run_uses_kex_pb_arguments(tmp_path):
    cx = _simulated_chemex(tmp_path)
    cx.run(residues=["G2", "A3"], kex=2000.0, kex_bounds=(100, 1e4),
           pb=0.02, pb_bounds=(0.0, 0.15), output_dir=tmp_path / "args")
    written = (tmp_path / "args" / "inputs" / "parameters.toml").read_text()
    assert "PB = [0.02, 0.0, 0.15]" in written and "KEX_AB = [2000.0, 100.0, 10000.0]" in written
    assert cx.results["initial"]["KEX_AB"] == [2000.0, 100.0, 10000.0]
    g = cx.global_parameters
    assert 100 <= g.loc["KEX_AB", "value"] <= 1e4 and 0 <= g.loc["PB", "value"] <= 0.15
    # a different start still converges near the truth (kex 800, pB 0.05)
    assert abs(g.loc["KEX_AB", "value"] - TRUE["KEX_AB"]) < 3 * g.loc["KEX_AB", "error"]


def test_default_method_fixes_cs_a(tmp_path):
    cx = _simulated_chemex(tmp_path)
    cx.run(output_dir=tmp_path / "cs")
    method = (tmp_path / "cs" / "inputs" / "method.toml").read_text()
    assert 'FIT = ["PB", "KEX_AB", "DW_AB"]' in method
    fitted = set(cx.parameters_table.query("kind == 'fitted'")["param"])
    assert fitted == {"PB", "KEX_AB", "DW_AB", "R2_A"}           # R2_A: ChemEx's own
    fixed = cx.parameters_table.query("kind == 'fixed' and param == 'CS_A'")
    assert dict(zip(fixed["spin"], fixed["value"])) == SHIFTS       # held at the peaks
    assert not list((tmp_path / "cs").rglob("*.pdf"))              # ChemEx PDFs are off


def test_method_with_cs_a_floats_within_window(tmp_path):
    cx = _simulated_chemex(tmp_path)
    cx.method = ["PB", "KEX_AB", "CS_A", "DW_AB"]
    cx.run(output_dir=tmp_path / "cs")
    params = (tmp_path / "cs" / "inputs" / "parameters.toml").read_text()
    assert "G2N = [115.0, 114.5, 115.5]" in params
    r = cx.residue_parameters
    assert ((r["CS_A"] - pd.Series(SHIFTS)).abs() <= 0.5 + 1e-9).all()
    with pytest.raises(ValueError, match="cs_window"):
        cx.run(cs_window=0)


def test_exchange_summary_then_global(tmp_path):
    """The hands-on workflow: individual fits -> summary -> global fit."""
    cx = _simulated_chemex(tmp_path)
    residues = ["G2", "A3"]
    with pytest.raises(RuntimeError, match="individual"):
        cx.exchange_summary()

    cx.run(residues=residues, fit="individual", output_dir=tmp_path / "ind")
    s = cx.exchange_summary()
    assert list(s.index) == ["kex", "pb"] and (s["n"] == 2).all()
    r = cx.residue_parameters
    w = 1 / r["KEX_AB_err"] ** 2
    assert s.loc["kex", "mean"] == pytest.approx((w * r["KEX_AB"]).sum() / w.sum())
    assert s.loc["kex", "min"] <= s.loc["kex", "mean"] <= s.loc["kex", "max"]
    assert s.loc["kex", "chi2_dof"] >= 0

    kex0 = min(max(s.loc["kex", "mean"], 15.0), 9e3)    # keep inside the default bounds
    pb0 = min(s.loc["pb", "mean"], 0.19)
    cx.run(residues=residues, fit="global", kex=kex0, pb=pb0, output_dir=tmp_path / "glob")
    assert cx.results["initial"]["KEX_AB"][0] == pytest.approx(kex0)
    g = cx.global_parameters
    assert abs(g.loc["KEX_AB", "value"] - TRUE["KEX_AB"]) < 3 * g.loc["KEX_AB", "error"]
    with pytest.raises(RuntimeError, match="individual"):
        cx.exchange_summary()                               # last run was global


def test_non_whole_ncyc_is_an_error():
    """T = 0.03 s with 50 Hz steps gives 1.5 cycles: refuse rather than fit."""
    fit_results = pd.DataFrame({"ref_index": 0, "plane": [0, 1, 2],
                                "vol": [1.0, 0.8, 0.9], "vol_err": 0.01})
    ref_df = pd.DataFrame({"ref_index": [0], "assn_label": ["G2"]})
    with pytest.raises(ValueError, match="50 Hz -> 1.5"):
        utils.cpmg_profiles(fit_results, [50.0, 100.0], 0.03, ref_df=ref_df)
    prof = utils.cpmg_profiles(fit_results, [50.0, 100.0], 0.04, ref_df=ref_df)
    assert list(prof["G2N-H"]["ncyc"]) == [0.0, 2.0, 4.0]


def test_plot_individual_and_global(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cx = _simulated_chemex(tmp_path)
    with pytest.raises(RuntimeError, match="run"):
        cx.plot(show=False)

    cx.run(fit="individual", output_dir=tmp_path / "ind")
    fig = cx.plot(show=False)
    assert len(fig.axes) == 4                          # 2 residues x (residuals + profile)
    titles = [a.get_title() for a in fig.axes if a.get_title()]
    assert titles[0].startswith("G2N\nkex ") and "Δω" in titles[0]
    line = [a for a in fig.axes if a.lines and len(a.lines[-1].get_xdata()) > 2][0].lines[-1]
    assert list(line.get_xdata()) == sorted(line.get_xdata())   # fit drawn in νCPMG order
    assert len(cx.plot(residues=["A3"], show=False).axes) == 2
    plt.close("all")

    cx.run(fit="global", output_dir=tmp_path / "glob", plot=True)   # shows, returns cx
    fig = cx.plot(show=False)
    assert fig._suptitle.get_text().startswith("global fit: kex ")
    plt.close("all")


def test_plot_matches_last_run_residues(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cx = _simulated_chemex(tmp_path)
    cx.run(residues=["G2"], fit="individual", output_dir=tmp_path / "one")
    assert len(cx.plot(show=False).axes) == 2           # only the residue that was fitted
    with pytest.raises(ValueError, match="A3N"):        # not in the last run: says so
        cx.plot(residues=["A3"], show=False)
    plt.close("all")


def test_plot_orders_panels_by_residue_number(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sim = _simulated_chemex(tmp_path)
    cx = ChemEx(dict(reversed(list(sim.profiles.items()))),   # A3 before G2
                EXPERIMENT, CONDITIONS, shifts=SHIFTS)
    assert cx.spins == ["G2N", "A3N"]
    cx.run(residues=["A3", "G2"], fit="global", output_dir=tmp_path / "glob")
    titles = [ax.get_title().split("\n")[0] for ax in cx.plot(show=False).axes
              if ax.get_title()]
    assert titles == ["G2N", "A3N"]
    plt.close("all")


def test_separate_r2_fits_r2b(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cx = _simulated_chemex(tmp_path)
    # default: R2_B tied to R2_A, so not fitted
    assert "R2_B" not in cx.build_method()["STEP1"]["ROLES"][0]["FIT"]
    assert cx.build_method(separate_r2=True)["STEP1"]["ROLES"][0]["FIT"][-1] == "R2_B"
    cx.run(fit="global", separate_r2=True, output_dir=tmp_path / "sep")
    method = (tmp_path / "sep" / "inputs" / "method.toml").read_text()
    assert "R2_B" in method
    rp = cx.residue_parameters
    assert {"R2_A", "R2_B"} <= set(rp.columns) and len(rp) == 2
    assert cx.results["separate_r2"] is True
    fig = cx.plot(show=False)
    assert any("R₂B" in ax.get_title() for ax in fig.axes)
    plt.close("all")
    # a method file can't be extended
    cx.method = tmp_path / "m.toml"
    with pytest.raises(ValueError, match="R2_B"):
        cx.build_method(separate_r2=True)
