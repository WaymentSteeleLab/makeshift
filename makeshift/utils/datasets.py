"""
Utilities for fetching datasets from Zenodo (or any URL).
"""

import hashlib
import urllib.request
from urllib.parse import urlparse
import zipfile
from pathlib import Path
from typing import Optional

# Registry of known datasets. Each entry:
#   url    : direct download URL (a .zip, or a single data file)
#   sha256 : hex digest of the downloaded file for integrity verification
DATASETS = {
    "SHP2_NSH2_CPMG": dict(
        url="https://zenodo.org/records/22161246/files/SHP2_NSH2_CPMG.zip",
        sha256="1a2e1e59a985a6ddac0dec56025c217cc0d22ddc0984c557c5038ca57755d0d7",
    ),
    "NTRC_HSQC": dict(
        url="https://zenodo.org/records/22161246/files/hsqcN15wfb_25deg_ntrc.ucsf",
        sha256="6ba340a740ad0d6aec678224e6d0040b1719185acbeb73823e36fb9abf11d35c",
    ),
}

_CACHE_DIR = Path.home() / ".makeshift" / "datasets"


def fetch(name: str, url: Optional[str] = None, sha256: Optional[str] = None,
          cache_dir: Optional[Path] = None, overwrite: bool = False) -> Path:
    """
    Download, verify, and cache a dataset. Zips are extracted; single files
    (e.g. a lone .ucsf spectrum) are downloaded as-is.

    Three ways to call it:

    * Registered dataset — pass a ``name`` from the DATASETS registry::

          fetch("SHP2_NSH2_CPMG")

    * Arbitrary dataset — pass a ``url`` (and optionally ``sha256``); ``name``
      is used as the cache folder::

          fetch("my_set", url="https://.../my_set.zip", sha256="...")

    * A URL directly as ``name`` — the cache folder is derived from the URL::

          fetch("https://.../my_set.zip")

    Parameters
    ----------
    name : str
        A registered dataset name, a cache-folder name (with ``url``), or a URL.
    url : str, optional
        Direct download URL. Overrides the registry; required for datasets not
        in DATASETS.
    sha256 : str, optional
        Expected hex digest for integrity verification. Falls back to the
        registry value for registered datasets; skipped if not available.
    cache_dir : Path, optional
        Root cache directory. Defaults to ~/.makeshift/datasets/.
    overwrite : bool
        If True, re-download even if already cached.

    Returns
    -------
    Path
        Directory containing the extracted dataset files, or the path to the
        downloaded file itself for a single-file dataset.
    """
    # allow a URL to be passed directly as `name`
    if url is None and _looks_like_url(name):
        url = name
        name = Path(urlparse(url).path).stem

    if url is None:
        # registry lookup
        if name not in DATASETS:
            available = list(DATASETS)
            raise ValueError(
                f"Unknown dataset {name!r} and no url given. "
                f"Registered: {available if available else '(none registered)'}. "
                f"Pass url=... to fetch an arbitrary dataset."
            )
        entry = DATASETS[name]
        url = entry["url"]
        if sha256 is None:
            sha256 = entry.get("sha256")

    root = Path(cache_dir) if cache_dir is not None else _CACHE_DIR
    extract_dir = root / name
    marker = extract_dir / ".complete"
    if marker.exists() and not overwrite:
        return _locate_data(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)
    fname = Path(urlparse(url).path).name
    dest_path = extract_dir / fname
    print(f"Downloading {name} from {url} ...")
    urllib.request.urlretrieve(url, dest_path, reporthook=_progress_hook())
    print()

    if sha256:
        actual = _sha256(dest_path)
        if actual != sha256:
            dest_path.unlink()
            raise RuntimeError(
                f"Checksum mismatch for {name}.\n"
                f"  expected: {sha256}\n"
                f"  got:      {actual}"
            )

    if dest_path.suffix == ".zip":
        print(f"Extracting to {extract_dir} ...")
        with zipfile.ZipFile(dest_path, "r") as zf:
            zf.extractall(extract_dir)
        dest_path.unlink()
    marker.touch()

    data_dir = _locate_data(extract_dir)
    print(f"Dataset {name!r} ready at {data_dir}")
    return data_dir


def list_datasets() -> list:
    """Return names of all registered datasets."""
    return list(DATASETS)


def _looks_like_url(s) -> bool:
    return isinstance(s, str) and s.startswith(("http://", "https://"))


def _locate_data(extract_dir: Path) -> Path:
    """Return the path holding the dataset. If there's a single top-level
    entry (a folder from zip extraction, or a lone downloaded file), return
    that directly; otherwise return extract_dir."""
    entries = [p for p in extract_dir.iterdir()
               if p.name != ".complete" and p.suffix != ".zip"]
    if len(entries) == 1:
        return entries[0]
    return extract_dir


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _progress_hook():
    """Return an urllib reporthook that prints a simple progress line."""
    def hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            mb = downloaded / 1e6
            total_mb = total_size / 1e6
            print(f"\r  {mb:.1f} / {total_mb:.1f} MB  ({pct:.0f}%)", end="", flush=True)
    return hook