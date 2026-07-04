import yaml
import urllib.request
from pathlib import Path

from ..spectra.spectrum import Spectrum


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

        time_T2: 0.05  # constant-time CPMG delay, seconds

        data_dir: /path/to/ucsf/files

        reference: ref.ucsf

        planes:
          - file: plane_0080.ucsf
            vcpmg: 80
          - file: plane_0120.ucsf
            vcpmg: 120
          ...

        sequence: MTEYKLVVVGA...   # optional, 1-indexed full construct sequence

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
        raise ValueError(f"Config is missing required keys: {missing}")
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
