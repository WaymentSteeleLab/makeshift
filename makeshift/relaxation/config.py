import yaml
import urllib.request
from pathlib import Path

from ..spectra.spectrum import Spectrum
from .acqus import cpmg_params_from_acqus


def _fetch_pdb(pdb_code: str) -> Path:
    """Download a PDB file from RCSB if not already cached, return its path."""
    cache_dir = Path.home() / ".makeshift" / "pdb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{pdb_code.upper()}.pdb"
    if not dest.exists():
        url = f"https://files.rcsb.org/download/{pdb_code.upper()}.pdb"
        print(f"  Downloading {pdb_code.upper()} from RCSB ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved → {dest}")
    return dest


def _yaml_set_cache(yaml_path: Path, cache_path: Path, key: str = "fit_lineshapes_cache") -> None:
    """Insert or replace a `<key>: <cache_path>` line in a YAML file."""
    text = yaml_path.read_text()
    new_line = f"{key}: {cache_path}\n"
    if f"{key}:" in text:
        lines = [
            new_line if line.startswith(f"{key}:") else line
            for line in text.splitlines(keepends=True)
        ]
        text = "".join(lines)
    else:
        text = text.rstrip("\n") + "\n" + new_line
    yaml_path.write_text(text)
    print(f"  Written {key} to {yaml_path.name}")


def load_config(yaml_path):
    """
    Load a CPMG experiment config from a YAML file.

    Expected format::

        time_T2: 0.04  # CPMG constant-time period T, s (always required)

        data_dir: /path/to/ucsf/files

        reference: ref.ucsf

        planes:
          - file: plane_0080.ucsf
            vcpmg: 80
          - file: plane_0120.ucsf
            vcpmg: 120
          ...

        sequence: MTEYKLVVVGA...   # optional, 1-indexed full construct sequence

        acqus: acqus               # optional Bruker acqus file (relative to
                                   # data_dir); supplies the ChemEx settings

        chemex:                    # optional, settings for CPMGExperiment.to_chemex
          experiment: cpmg_15n_ip  # ChemEx experiment name (default)
          model: 2st               # optional; also experiment_settings,
                                   # conditions, parameters, method
          # h_larmor_frq / carrier / temperature / pw90 come from acqus;
          # set them here only to override it (or when there is no acqus)

    With ``acqus``, ``pw90`` (p30 / 2), ``carrier``
    (O3 / BF3), ``h_larmor_frq`` (BF1), ``temperature`` (TE) and
    ``time_equil`` (d20) are read from it (see
    :mod:`makeshift.relaxation.acqus`); values in the ``chemex`` section
    override acqus. ``time_T2`` is never read from acqus (how T relates to
    the delays depends on the pulse program); d17 is printed for comparison.
    The resulting ChemEx settings are validated here (unknown keys, missing
    ``temperature`` / ``pw90``, non-numeric values raise ValueError) so a
    typo fails before the lineshape fit rather than after it; see
    :func:`makeshift.chemex.utils.normalize_settings`.

    Parameters
    ----------
    yaml_path : str or Path

    Returns
    -------
    dict with keys: time_T2, reference, planes
        reference and planes[*].file are resolved to absolute Path objects.
        'sequence', if present, is returned as-is (str).
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    required = {"time_T2", "data_dir", "reference", "planes"}
    missing = required - set(cfg.keys())
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}"
                         + (" (time_T2, the CPMG constant-time period in s, is not "
                            "read from acqus)" if "time_T2" in missing else ""))
    if not isinstance(cfg["planes"], list) or len(cfg["planes"]) == 0:
        raise ValueError("'planes' must be a non-empty list")
    for i, plane in enumerate(cfg["planes"]):
        if "file" not in plane or "vcpmg" not in plane:
            raise ValueError(f"planes[{i}] is missing 'file' or 'vcpmg'")

    data_dir = Path(cfg["data_dir"]).expanduser()
    cfg["reference"] = data_dir / cfg["reference"]
    for plane in cfg["planes"]:
        plane["file"] = data_dir / plane["file"]
    if "pdb" in cfg:
        pdb_val = str(cfg["pdb"]).strip()
        if len(pdb_val) == 4 and pdb_val.isalnum():
            cfg["pdb"] = _fetch_pdb(pdb_val)
        else:
            cfg["pdb"] = Path(pdb_val).expanduser()

    cfg.setdefault("peak_mapping_tol", 1)
    cfg.setdefault("baseline_ref_plane", 10)
    cfg.setdefault("start_num", 1)
    cfg.setdefault("end_num", 3)
    acq = None
    if "acqus" in cfg:
        acqus_path = Path(cfg["acqus"]).expanduser()
        if not acqus_path.is_absolute():
            acqus_path = data_dir / acqus_path
        acq = cpmg_params_from_acqus(acqus_path)
        cfg["acqus"] = acqus_path
        cfg["acqus_params"] = acq
        d17 = f"{acq['d17'] * 1e3:g} ms" if acq.get("d17") is not None else "n/a"
        print(f"  acqus ({acq['pulse_program']}): pw90 = {acq['pw90'] * 1e6:g} us, "
              f"carrier = {acq['carrier']:.3f} ppm, 1H = {acq['h_larmor_frq']:.2f} MHz, "
              f"temperature = {acq['temperature']:.2f} C, time_equil = "
              f"{acq['time_equil'] * 1e3:g} ms; d17 = {d17} (not used)")
        print(f"  time_T2 = {float(cfg['time_T2']) * 1e3:g} ms (from the config)")

    if acq is not None or "chemex" in cfg:
        cfg["chemex"] = _chemex_settings(cfg.get("chemex"), acq,
                                         source=f"{yaml_path}: chemex section")
    cfg["yaml_path"] = Path(yaml_path)

    return cfg


def load_planes(config):
    """
    Read all UCSF planes described in a config dict (from load_config).

    Parameters
    ----------
    config : dict
        Output of load_config.

    Returns
    -------
    ref_spectrum : Spectrum — reference plane (data + unit-conversion)
    plane_data : list of ndarray — [ref_data, cpmg_plane_1, cpmg_plane_2, ...]
    vcpmg_values : list of float — νCPMG in Hz, one per CPMG plane
    time_T2 : float — constant-time delay in seconds
    """
    ref_spectrum = Spectrum.from_ucsf(config["reference"])
    sorted_planes = sorted(config["planes"], key=lambda p: float(p["vcpmg"]))
    plane_data = [ref_spectrum.data]
    vcpmg_values = []
    for plane in sorted_planes:
        plane_data.append(Spectrum.from_ucsf(plane["file"]).data)
        print(f'read data from {plane["file"]}')
        vcpmg_values.append(float(plane["vcpmg"]))
    return ref_spectrum, plane_data, vcpmg_values, float(config["time_T2"])


def _chemex_settings(section, acq, source):
    """ChemEx settings: values from acqus, overridden by the yaml's chemex
    section, then validated."""
    from ..chemex.utils import normalize_settings
    if section is not None and not isinstance(section, dict):
        raise ValueError(f"{source} must be a mapping")
    merged = {}
    if acq is not None:
        merged = {k: acq[k] for k in ("h_larmor_frq", "carrier", "temperature", "pw90")}
        merged["experiment_settings"] = {"time_equil": acq["time_equil"]}
    for key, value in (section or {}).items():
        if key == "experiment_settings" and isinstance(value, dict):
            merged[key] = {**merged.get(key, {}), **value}
        else:
            merged[key] = value
    return normalize_settings(merged, source=source)
