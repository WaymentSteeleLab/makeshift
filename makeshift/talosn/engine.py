"""
TALOS-N engine: the :class:`TalosN` object plus the environment helpers that
download and locate the NIH TALOS-N binary.

:class:`TalosN` holds a long-format shift table and the polymer sequence and,
on :meth:`TalosN.run`, writes a TALOS-N input ``.tab`` (via :mod:`.utils`),
invokes the binary, and parses the output tables back into DataFrames.
Stateless format/column helpers live in :mod:`makeshift.talosn.utils`.

The database/weight files and platform binary are downloaded on demand. By
default they land in ``makeshift/talosn/talosn_data``, but every entry point
takes a ``data_dir`` argument so you can install and run from anywhere; keep the
path in a variable and pass the same one to install and to :class:`TalosN`.
"""

import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

from . import utils

_PACKAGE_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = _PACKAGE_DIR / "talosn_data"

TALOSN_DOWNLOAD_URL = (
    "https://spin.niddk.nih.gov/bax-apps/software/TALOS-N/talosn.tZ"
)
TALOSN_INFO_URL = "https://spin.niddk.nih.gov/bax-apps/software/TALOS-N/"
TALOSN_TERMS_URL = "https://spin.niddk.nih.gov/bax-apps/terms.html"

TALOSN_TERMS_NOTICE = (
    "A stable version TALOS-N software package can be downloaded below. When \n"
    "downloading software from this website, you are agreeing to our Terms of \n"
    "Use, including the terms that there is no right to privacy on this system, \n"
    "and that the software from this website is not to be redistributed without \n"
    "permission from the authors. The TALOS-N package provides the hardware & \n"
    "OS versions of linux, linux9, winxp and mac, and requires at least ~0.5 GB \n"
    "memory (to load the required library).\n"
    f"Terms of Use: {TALOSN_TERMS_URL}"
)

_MIN_TALOS_TAB_BYTES = 10_000_000


def _resolve_data_dir(data_dir=None):
    """The TALOS-N data root: ``data_dir`` if given, else the package default."""
    return Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR

def is_talosn_data_installed(data_dir=None):
    talos_tab = _resolve_data_dir(data_dir) / "tab" / "talos.tab"
    if not talos_tab.is_file():
        return False
    return talos_tab.stat().st_size >= _MIN_TALOS_TAB_BYTES


def _fix_tab_symlinks(data_dir=None):
    # TALOS-N binary expects talos_obsCS.tab; NIH package ships talos.obsCS.tab
    tab_dir = _resolve_data_dir(data_dir) / "tab"
    tab_dir.mkdir(parents=True, exist_ok=True)
    obs_dot = tab_dir / "talos.obsCS.tab"
    obs_under = tab_dir / "talos_obsCS.tab"
    if obs_dot.is_file() and not obs_under.exists():
        obs_under.symlink_to("talos.obsCS.tab")


def install_talosn_data(data_dir=None, force=False, install_binaries=True,
                        url=TALOSN_DOWNLOAD_URL):

    data_dir = _resolve_data_dir(data_dir)
    tab_dir = data_dir / "tab"
    bin_dir = data_dir / "bin"

    try:
        binary_installed = _get_talosn_binary(data_dir)
    except RuntimeError:
        binary_installed = None

    if is_talosn_data_installed(data_dir) and binary_installed and not force and not install_binaries:
        _fix_tab_symlinks(data_dir)
        return data_dir

    tab_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    archive_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".tZ", delete=False) as tmp:
            archive_path = Path(tmp.name)

        print(TALOSN_TERMS_NOTICE)
        print(f"\nDownloading TALOS-N data from {url} ...")
        urllib.request.urlretrieve(url, archive_path)

        with tarfile.open(archive_path, "r:*") as tar:
            for member in tar.getmembers():
                if member.name.startswith("tab/") and member.isfile():
                    dest = tab_dir / Path(member.name).name
                    if force or not dest.exists():
                        tar.extract(member, path=data_dir)
                elif install_binaries and member.name.startswith("bin/") and member.isfile():
                    dest = bin_dir / Path(member.name).name
                    if force or not dest.exists():
                        tar.extract(member, path=data_dir)
                    if dest.is_file():
                        dest.chmod(dest.stat().st_mode | 0o111)

        _fix_tab_symlinks(data_dir)

        if not is_talosn_data_installed(data_dir):
            raise RuntimeError(
                "TALOS-N data installation finished but talos.tab was not found. "
                f"Try downloading manually from {TALOSN_INFO_URL}"
            )

        print(f"TALOS-N data installed in {data_dir}")
        return data_dir

    finally:
        if archive_path is not None and archive_path.exists():
            archive_path.unlink()


def ensure_talosn_data(data_dir=None, auto_install=False):
    data_dir = _resolve_data_dir(data_dir)
    _fix_tab_symlinks(data_dir)
    if is_talosn_data_installed(data_dir):
        return
    if auto_install:
        install_talosn_data(data_dir)
        return
    raise RuntimeError(
        f"TALOS-N database/weight files are not installed in {data_dir}. "
        "Run makeshift.talosn.install_talosn_data() to download them from NIH, "
        f"or see {TALOSN_INFO_URL}"
    )


def _detect_platform(bin_dir):
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        # TALOSN.mac.M segfaults on some Apple Silicon setups
        if (bin_dir / "TALOSN.mac.10.15").exists():
            return "TALOSN.mac.10.15"
        if machine in ("arm64", "aarch64") and (bin_dir / "TALOSN.mac.M").exists():
            return "TALOSN.mac.M"
        return "TALOSN.mac"

    if system == "linux":
        if machine in ("x86_64", "amd64"):
            if (bin_dir / "TALOSN.static.linux9_x64").exists():
                return "TALOSN.static.linux9_x64"
            return "TALOSN.linux9_x64"
        if (bin_dir / "TALOSN.static.linux9").exists():
            return "TALOSN.static.linux9"
        return "TALOSN.linux9"

    if system == "windows":
        return "TALOSN.win"

    raise RuntimeError(f"Unsupported platform: {system} {machine}")


def _get_talosn_binary(data_dir=None):
    bin_dir = _resolve_data_dir(data_dir) / "bin"
    preferred = _detect_platform(bin_dir)
    candidates = [preferred]
    alternatives = {
        "TALOSN.mac.10.15": ["TALOSN.mac.M", "TALOSN.mac"],
        "TALOSN.mac.M": ["TALOSN.mac.10.15", "TALOSN.mac"],
        "TALOSN.mac": ["TALOSN.mac.10.15", "TALOSN.mac.M"],
        "TALOSN.linux9_x64": ["TALOSN.static.linux9_x64", "TALOSN.linux9"],
        "TALOSN.static.linux9_x64": ["TALOSN.linux9_x64", "TALOSN.static.linux9"],
        "TALOSN.linux9": ["TALOSN.static.linux9", "TALOSN.linux9_x64"],
        "TALOSN.static.linux9": ["TALOSN.linux9", "TALOSN.static.linux9_x64"],
    }
    candidates.extend(alternatives.get(preferred, []))

    for name in candidates:
        path = bin_dir / name
        if path.is_file():
            return path

    raise RuntimeError(
        f"No TALOS-N binary found in {bin_dir}. "
        "Install binaries with install_talosn_data(install_binaries=True)."
    )


def _run_talosn(input_tab, output_dir, data_dir=None, reference_pdb=None,
                auto_exclude=True, no_proton=False, extra_args=None):
    data_dir = _resolve_data_dir(data_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_tab = Path(input_tab)

    if input_tab.parent.resolve() == output_dir.resolve():
        input_arg = input_tab.name
    else:
        input_arg = str(input_tab.resolve())

    binary = _get_talosn_binary(data_dir)
    cmd = [
        str(binary),
        "-in", input_arg,
        "-talosnDir", str(data_dir.resolve()),
    ]
    if reference_pdb:
        cmd.extend(["-ref", str(reference_pdb)])
    if auto_exclude:
        cmd.append("-autoExcl")
    if no_proton:
        cmd.append("-noproton")
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env["TALOSN_DIR"] = str(data_dir.resolve())

    return subprocess.run(cmd, cwd=str(output_dir), env=env,
                          capture_output=True, text=True)


def _collect_output_files(output_dir):
    output_dir = Path(output_dir)
    known_outputs = {
        "predictions": "pred.tab",
        "predictions_all": "predAll.tab",
        "adjusted_shifts": "predAdjCS.tab",
        "s2": "predS2.tab",
        "secondary_structure": "predSS.tab",
        "chi1": "predChi1.tab",
        "abp": "predABP.tab",
    }

    results = {"output_dir": str(output_dir)}
    for key, filename in known_outputs.items():
        path = output_dir / filename
        results[f"{key}_file"] = str(path) if path.is_file() else None
        if path.is_file():
            try:
                results[key] = utils.parse_tab(path)
            except (ValueError, pd.errors.ParserError):
                results[key] = None
    return results

class TalosN:
    """
    Predict backbone torsion angles, S2 order parameters, and secondary
    structure from assigned chemical shifts using the NIH TALOS-N binary.

    Build from a shift table:

        tn = TalosN.from_bmrb(15000, data_dir="~/talosn_data")
        tn.run(auto_install=True)
        tn.order_parameters        # predS2 table
        tn.torsion_angles          # pred.tab
        tn.secondary_structure     # predSS.tab

    """

    def __init__(self, shifts, sequence=None, entry_id=None, entity_id=None,
                 entry=None, data_dir=None, first_resid=None):
        self.shifts = shifts
        self.sequence = sequence
        self.entry_id = entry_id
        self.entity_id = entity_id
        self.entry = entry
        self.data_dir = data_dir
        self.first_resid = first_resid
        self.results = None

    @classmethod
    def from_bmrb(cls, bmrb_id, entity_id=None, sequence=None, data_dir=None, **fetch_kw):
        """Fetch a BMRB entry and build from its assigned chemical shifts."""
        from ..entry import NMRStarEntry
        entry = NMRStarEntry.from_bmrb(bmrb_id, **fetch_kw)
        return cls.from_entry(entry, entity_id=entity_id, sequence=sequence,
                              data_dir=data_dir)

    @classmethod
    def from_entry(cls, entry, entity_id=None, sequence=None, data_dir=None):
        """Build from an already-parsed :class:`NMRStarEntry`."""
        from ..chemshift import ChemicalShifts

        if sequence is None:
            seqs = entry.sequences()
            if entity_id is not None:
                sequence = entry.sequences(entity_id=entity_id)
            else:
                poly = seqs[
                    seqs["Polymer_type"].str.contains("polypeptide", case=False, na=False)
                ]
                if poly.empty:
                    raise ValueError(
                        f"No polypeptide sequence found in entry {getattr(entry, 'entry_id', None)}"
                    )
                sequence = poly.iloc[0]["Polymer_seq_one_letter_code"]
                entity_id = poly.iloc[0]["ID"]
            if not isinstance(sequence, str) or not sequence or pd.isna(sequence):
                raise ValueError("could not resolve a sequence; pass sequence=... explicitly")

        shifts = utils.filter_backbone(ChemicalShifts.from_entry(entry).data)
        if shifts.empty:
            raise ValueError(
                f"No backbone chemical shifts in entry {getattr(entry, 'entry_id', None)}"
            )
        first_resid = entry.resolve_first_resid(entity_id, sequence, shifts)
        return cls(shifts, sequence,
                   entry_id=getattr(entry, "entry_id", None),
                   entity_id=entity_id, entry=entry, data_dir=data_dir,
                   first_resid=first_resid)

    def run(self, output_dir=None, reference_pdb=None, auto_exclude=True,
            no_proton=False, cleanup=False, auto_install=False):

        ensure_talosn_data(self.data_dir, auto_install=auto_install)

        temp_dir = None
        if output_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="talosn_")
            output_dir = temp_dir
        else:
            output_dir = str(output_dir)
            os.makedirs(output_dir, exist_ok=True)

        try:
            if isinstance(self.shifts, pd.DataFrame):
                cs_df = utils.filter_backbone(self.shifts)
                input_file = os.path.join(output_dir, "input.tab")
                utils.shifts_to_tab(cs_df, input_file, sequence=self.sequence,
                                     first_resid=self.first_resid)
            else:
                input_file = str(self.shifts)
                if not os.path.exists(input_file):
                    raise FileNotFoundError(f"Input file not found: {input_file}")

            proc = _run_talosn(
                input_tab=input_file, output_dir=output_dir, data_dir=self.data_dir,
                reference_pdb=reference_pdb, auto_exclude=auto_exclude,
                no_proton=no_proton,
            )

            outputs = _collect_output_files(Path(output_dir))
            outputs["returncode"] = proc.returncode
            outputs["stdout"] = proc.stdout
            outputs["stderr"] = proc.stderr

            if proc.returncode != 0 and outputs.get("predictions") is None:
                raise RuntimeError(
                    f"TALOS-N failed with return code {proc.returncode}.\n"
                    f"stderr (last 2000 chars):\n{proc.stderr[-2000:]}"
                )

            self.results = outputs
            return self

        except Exception:
            if cleanup and temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise

    def predict_s2(self, **run_kw):
        """Run if needed and return the predS2 table, raising if absent."""
        if self.results is None:
            self.run(**run_kw)
        s2_df = self.results.get("s2")
        if s2_df is None or len(s2_df) == 0:
            raise RuntimeError(
                "TALOS-N did not produce predS2.tab. Ensure backbone carbon "
                "shifts (CA/CB/C) are present and that TALOS-N data are "
                "installed via install_talosn_data()."
            )
        return s2_df

    @property
    def torsion_angles(self):
        return None if self.results is None else self.results.get("predictions")

    @property
    def order_parameters(self):
        return None if self.results is None else self.results.get("s2")

    @property
    def secondary_structure(self):
        return None if self.results is None else self.results.get("secondary_structure")

    def __repr__(self):
        n = self.shifts["Seq_ID"].nunique() if (
            isinstance(self.shifts, pd.DataFrame) and "Seq_ID" in self.shifts.columns
        ) else "?"
        state = "run" if self.results is not None else "not run"
        return f"<TalosN entry={self.entry_id} residues={n} ({state})>"