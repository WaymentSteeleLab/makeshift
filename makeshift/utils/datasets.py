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
#   url    : direct download URL for the zip
#   sha256 : hex digest of the zip for integrity verification
DATASETS = {
    "SHP2_NSH2_CPMG": dict(
        url="https://zenodo.org/records/20732240/files/SHP2_NSH2_CPMG.zip",
        sha256="1a2e1e59a985a6ddac0dec56025c217cc0d22ddc0984c557c5038ca57755d0d7",
    ),
}

_CACHE_DIR = Path.home() / ".makeshift" / "datasets"


def fetch(name: str, url: Optional[str] = None, sha256: Optional[str] = None,
          cache_dir: Optional[Path] = None, overwrite: bool = False) -> Path:
    """
    Download, verify, cache, and extract a dataset zip.

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
        Directory containing the extracted dataset files.
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
    zip_path = extract_dir / f"{name}.zip"
    print(f"Downloading {name} from {url} ...")
    urllib.request.urlretrieve(url, zip_path, reporthook=_progress_hook())
    print()

    if sha256:
        actual = _sha256(zip_path)
        if actual != sha256:
            zip_path.unlink()
            raise RuntimeError(
                f"Checksum mismatch for {name}.\n"
                f"  expected: {sha256}\n"
                f"  got:      {actual}"
            )

    print(f"Extracting to {extract_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    zip_path.unlink()
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
    """Return the directory holding the dataset. If the zip extracted to a
    single top-level folder, descend into it; otherwise return extract_dir."""
    entries = [p for p in extract_dir.iterdir()
               if p.name != ".complete" and p.suffix != ".zip"]
    dirs = [p for p in entries if p.is_dir()]
    if len(entries) == 1 and len(dirs) == 1:
        return dirs[0]
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