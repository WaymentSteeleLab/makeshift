# Datasets & structures

`makeshift.utils` bundles two dependency-light fetchers — one for example
**datasets**, one for protein **structures** — plus a set of constants. Both
cache under `~/.makeshift/`.

## Datasets

```python
from makeshift.utils import datasets

datasets.list_datasets()          # registered dataset names
path = datasets.fetch("SHP2_NSH2_CPMG")   # download, verify, cache, extract
```

`fetch` downloads a dataset zip, verifies its checksum, caches it, and extracts
it — returning the local path. It's how the [CPMG example](cpmg.md) gets its
`.ucsf` planes. You can also point it at a custom `url` with an expected
`sha256`.

## Structures

`fetch_structure` returns a local path to a PDB structure, downloading it if
needed. It infers the source from the identifier, or you can force it:

```python
from makeshift.utils import fetch_structure

fetch_structure("model.pdb")     # local file — returned as-is
fetch_structure("1WRP")          # 4-char PDB id  -> RCSB
fetch_structure("P0DP23")        # UniProt accession -> AlphaFold DB
fetch_structure("P0DP23", source="afdb", version="v4")
```

| `source` | Fetches from |
|---|---|
| `"auto"` (default) | inferred from the identifier |
| `"file"` | local path |
| `"rcsb"` | RCSB PDB |
| `"afdb"` | AlphaFold DB |

`detect_source(identifier)` exposes the inference on its own if you just want to
know where an id would resolve.

These feed directly into [HYDRONMR](hydronmr.md) and the rigid-body step of a
[relaxation profile](relaxation.md).

## Constants and reference tables

`makeshift.utils.constants` holds amino-acid code maps, backbone atom lists,
labelling keywords, and identifier regexes used throughout the package. The
bundled reference tables (random-coil shifts, PANAV distributions, BMRB
statistics) are accessible via `makeshift.data.tables`.

## Full API

See the [Utilities reference](../api/utils.md).
