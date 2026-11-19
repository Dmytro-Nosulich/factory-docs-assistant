"""The documents in the stage 2 RAG index, as listed in data/manifest.csv (decision 14)."""

import csv
from pathlib import Path

from factory_docs.config import ROOT

MANIFEST = ROOT / "data" / "manifest.csv"


def in_scope(manifest: Path = MANIFEST) -> list[dict[str, str]]:
    """Return the manifest rows of every document in the RAG index, in manifest order."""
    with manifest.open(encoding="utf-8", newline="") as f:
        return [row for row in csv.DictReader(f) if row["in_rag_index"] == "yes"]
