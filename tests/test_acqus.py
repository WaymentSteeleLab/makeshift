"""Reading CPMG parameters from a Bruker acqus file, and using it in the config."""

import pytest

from makeshift.relaxation.acqus import cpmg_params_from_acqus, read_acqus
from makeshift.relaxation.config import load_config


def _acqus_text(d17=0.04, d20=0.005, p30=70.0, o3=-1033.98, bf3=60.81,
                bf1=600.13, te=298.15):
    """A Bruker-style acqus: arrays over several lines, as the spectrometer writes."""
    d = ["0"] * 64
    d[1], d[17], d[20] = "1.5", repr(d17), repr(d20)
    p = ["0"] * 64
    p[1], p[30] = "9.5", repr(p30)
    rows = lambda vals: "\n".join(" ".join(vals[i:i + 8]) for i in range(0, 64, 8))
    return (
        "##TITLE= Parameter file, TopSpin 4.1.4\n"
        "##JCAMPDX= 5.0\n"
        "$$ /opt/data/cpmg/3/acqus\n"
        "##$BF1= 600.13\n".replace("600.13", repr(bf1)) +
        f"##$BF3= {bf3}\n"
        f"##$D= (0..63)\n{rows(d)}\n"
        "##$NS= 16\n"
        f"##$O3= {o3}\n"
        f"##$P= (0..63)\n{rows(p)}\n"
        "##$PULPROG= <N15CPMG.wtg.NEW_tst1>\n"
        f"##$TE= {te}\n"
        "##END=\n"
    )


def test_read_acqus_parses_arrays_strings_and_scalars(tmp_path):
    path = tmp_path / "acqus"
    path.write_text(_acqus_text())
    par = read_acqus(path)
    assert len(par["D"]) == 64 and par["D"][17] == 0.04 and par["D"][1] == 1.5
    assert len(par["P"]) == 64 and par["P"][30] == 70.0
    assert par["PULPROG"] == "N15CPMG.wtg.NEW_tst1"
    assert par["BF1"] == 600.13 and par["NS"] == 16.0


def test_cpmg_params_match_reference_mapping(tmp_path):
    path = tmp_path / "acqus"
    path.write_text(_acqus_text())
    p = cpmg_params_from_acqus(path)
    assert "time_T2" not in p                                # T is never read from acqus
    assert p["d17"] == pytest.approx(0.04)                   # only for comparison
    assert p["time_equil"] == pytest.approx(0.005)
    assert p["pw90"] == pytest.approx(35e-6)                 # p30 / 2, us -> s
    assert p["carrier"] == pytest.approx(-1033.98 / 60.81)   # O3 / BF3, ppm
    assert p["h_larmor_frq"] == pytest.approx(600.13)
    assert p["temperature"] == pytest.approx(25.0)           # K -> C
    assert p["pulse_program"] == "N15CPMG.wtg.NEW_tst1"


def test_cpmg_params_reject_zero_pulse(tmp_path):
    path = tmp_path / "acqus"
    path.write_text(_acqus_text(p30=0.0))
    with pytest.raises(ValueError, match="pw90"):
        cpmg_params_from_acqus(path)


BASE = """\
data_dir: {data_dir}
reference: ref.ucsf
planes:
  - {{file: p50.ucsf, vcpmg: 50}}
"""


def test_config_reads_chemex_settings_from_acqus(tmp_path):
    (tmp_path / "acqus").write_text(_acqus_text(d17=0.03))
    cfg_path = tmp_path / "cpmg.yaml"

    # time_T2 is still required with acqus ...
    cfg_path.write_text(BASE.format(data_dir=tmp_path) + "acqus: acqus\n")
    with pytest.raises(ValueError, match="time_T2"):
        load_config(cfg_path)

    # ... and comes from the config, never from d17; no chemex section needed
    cfg_path.write_text(BASE.format(data_dir=tmp_path) + "acqus: acqus\ntime_T2: 0.04\n")
    cfg = load_config(cfg_path)
    assert cfg["time_T2"] == 0.04 and cfg["acqus_params"]["d17"] == pytest.approx(0.03)
    assert cfg["acqus"] == tmp_path / "acqus"
    cx = cfg["chemex"]
    assert cx["pw90"] == pytest.approx(35e-6) and cx["temperature"] == pytest.approx(25.0)
    assert cx["h_larmor_frq"] == pytest.approx(600.13)
    assert cx["carrier"] == pytest.approx(-1033.98 / 60.81)
    assert cx["experiment_settings"] == {"time_equil": pytest.approx(0.005)}

    # the chemex section overrides acqus; experiment_settings merge
    cfg_path.write_text(BASE.format(data_dir=tmp_path) +
                        "acqus: acqus\ntime_T2: 0.04\nchemex:\n  temperature: 30\n"
                        "  experiment: cpmg_15n_tr\n"
                        "  experiment_settings: {antitrosy: false}\n")
    cx = load_config(cfg_path)["chemex"]
    assert cx["temperature"] == 30.0 and cx["experiment"] == "cpmg_15n_tr"
    assert cx["pw90"] == pytest.approx(35e-6)
    assert cx["experiment_settings"] == {"time_equil": pytest.approx(0.005),
                                         "antitrosy": False}



def test_config_without_acqus_still_needs_time_T2(tmp_path):
    cfg_path = tmp_path / "cpmg.yaml"
    cfg_path.write_text(BASE.format(data_dir=tmp_path))
    with pytest.raises(ValueError, match="time_T2"):
        load_config(cfg_path)
    cfg_path.write_text(BASE.format(data_dir=tmp_path) + "time_T2: 0.04\n")
    cfg = load_config(cfg_path)
    assert cfg["time_T2"] == 0.04 and "chemex" not in cfg
