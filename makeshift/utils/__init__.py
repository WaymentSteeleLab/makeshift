"""Dependency-light helpers: dataset and structure fetching, plus constants."""

from .structures import fetch_structure, detect_source
from .datasets import fetch, list_datasets

__all__ = [
    "fetch_structure",
    "detect_source",
    "fetch",
    "list_datasets",
]
