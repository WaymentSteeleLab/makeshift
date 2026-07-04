"""
Fetch a protein structure by identifier, from a local file, RCSB PDB, or the
AlphaFold Protein Structure Database (AFDB).
"""

import os
import re
import urllib.request
import urllib.error

from ..utils.constants import _UNIPROT_RE

_RCSB_URL = "https://files.rcsb.org/download/{id}.pdb"
_AFDB_URL = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F{frag}-model_{version}.pdb"

def _default_cache():
    return os.path.join(os.path.expanduser("~"), ".makeshift", "structures")


def detect_source(identifier):
    """
    Infer where a structure identifier should be fetched from:
        'file' if it's an existing path
        'rcsb' for a PDB id, 
        'afdb' for a UniProt accession
    Raises if it matches neither pattern (pass source= explicitly).
    """
    if os.path.exists(identifier):
        return "file"
    s = identifier.strip()
    if _PDB_RE.match(s):
        return "rcsb"
    if _UNIPROT_RE.match(s.upper()):
        return "afdb"
    raise ValueError(
        f"could not infer a source for {identifier!r}; "
        "pass source='file', 'rcsb', or 'afdb'"
    )


def fetch_structure(identifier, source="auto", version="v6", frag=1,
                    cache_dir=None, overwrite=False):
    """
    Return a local path to a PDB structure, downloading it if needed.

    Parameters
    ----------
    identifier : str
        A local file path, a 4-character PDB id (RCSB), or a UniProt accession
        (AlphaFold DB).
    source : {'auto', 'file', 'rcsb', 'afdb'}
        Where to fetch from; 'auto' infers it from the identifier.
    version : str
        AFDB model version (default 'v4'; its coordinates are carried forward by
        later AFDB releases, so this stays the stable default).
    frag : int
        AFDB fragment number (1 for all but very long proteins).
    cache_dir : str or None
    """
    source = source.lower()
    if source == "auto":
        source = detect_source(identifier)

    if source == "file":
        if not os.path.exists(identifier):
            raise FileNotFoundError(identifier)
        return identifier

    cache_dir = cache_dir or _default_cache()
    os.makedirs(cache_dir, exist_ok=True)

    if source == "rcsb":
        pid = identifier.strip().lower()
        url = _RCSB_URL.format(id=pid)
        dest = os.path.join(cache_dir, f"{pid}.pdb")
    elif source == "afdb":
        acc = identifier.strip().upper()
        url = _AFDB_URL.format(acc=acc, frag=frag, version=version)
        dest = os.path.join(cache_dir, f"AF-{acc}-F{frag}-{version}.pdb")
    else:
        raise ValueError(f"unknown source {source!r}. Should be auto, file, rcsb, or afdb")

    if os.path.exists(dest) and not overwrite:
        return dest
    try:
        urllib.request.urlretrieve(url, dest)
    except urllib.error.HTTPError as e:
        if os.path.exists(dest):
            os.remove(dest)
        raise ValueError(
            f"could not fetch {identifier!r} from {source} ({url}): HTTP {e.code}"
        ) from e
    return dest