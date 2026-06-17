"""
Utilities for fetching example CPMG datasets.

Datasets are hosted on Zenodo. On first call, files are downloaded to a local
cache directory (~/.makeshift/datasets/<dataset_name>/) and reused on
subsequent calls.

Usage
-----
>>> import makeshift as ms
>>> data_dir = ms.datasets.fetch("SHP2_NSH2_CPMG")
"""

import hashlib
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Registry — fill in doi/url/sha256 after uploading to Zenodo
# ---------------------------------------------------------------------------

# Each entry:
#   url    : direct download URL for the zip on Zenodo
#   sha256 : hex digest of the zip for integrity verification
DATASETS = {
    "SHP2_NSH2_CPMG": dict(
        url="https://zenodo.org/records/20732240/files/SHP2_NSH2_CPMG.zip",
        sha256="1a2e1e59a985a6ddac0dec56025c217cc0d22ddc0984c557c5038ca57755d0d7",
    ),
}

_CACHE_DIR = Path.home() / ".makeshift" / "datasets"


def fetch(name: str, cache_dir: Optional[Path] = None, overwrite: bool = False) -> Path:
    """
    Download and cache an example CPMG dataset by name.

    Parameters
    ----------
    name : str
        Dataset name (key in the DATASETS registry).
    cache_dir : Path, optional
        Root cache directory. Defaults to ~/.makeshift/datasets/.
    overwrite : bool
        If True, re-download even if the dataset is already cached.

    Returns
    -------
    Path
        Directory containing the dataset .ucsf files. The corresponding YAML
        config for use with makeshift.cpmg.run_protein() is provided in the
        makeshift repository under examples/.
    """
    if name not in DATASETS:
        available = list(DATASETS)
        raise ValueError(
            f"Unknown dataset {name!r}. "
            f"Available: {available if available else '(none registered yet)'}"
        )

    root = Path(cache_dir) if cache_dir is not None else _CACHE_DIR
    extract_dir = root / name
    data_dir = extract_dir / name
    marker = extract_dir / ".complete"

    if marker.exists() and not overwrite:
        return data_dir

    entry = DATASETS[name]
    url = entry["url"]
    expected_sha256 = entry.get("sha256")

    extract_dir.mkdir(parents=True, exist_ok=True)
    zip_path = extract_dir / f"{name}.zip"

    print(f"Downloading {name} from {url} ...")
    urllib.request.urlretrieve(url, zip_path, reporthook=_progress_hook())
    print()

    if expected_sha256:
        actual = _sha256(zip_path)
        if actual != expected_sha256:
            zip_path.unlink()
            raise RuntimeError(
                f"Checksum mismatch for {name}.\n"
                f"  expected: {expected_sha256}\n"
                f"  got:      {actual}"
            )

    print(f"Extracting to {extract_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    zip_path.unlink()

    marker.touch()
    print(f"Dataset {name!r} ready at {data_dir}")
    return data_dir


def list_datasets() -> list:
    """Return names of all registered datasets."""
    return list(DATASETS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
