"""Search the index from the command line (S2-06: vector search only).

Usage: uv run python scripts/search.py "question" [--collection structure|fixed] [--k 5]
"""

import argparse

from qdrant_client import QdrantClient

from factory_docs.config import ROOT, load_config
from factory_docs.embedding import FastEmbedder
from factory_docs.index import search

SNIPPET = 150


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("--collection", default="structure", choices=list(config.index.collections))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    store = ROOT / config.index.path
    if not store.exists():
        raise SystemExit(f"No index at {config.index.path}; run `make index` first")
    embedder = FastEmbedder.from_config(config)
    client = QdrantClient(path=str(store))
    try:
        hits = search(client, args.collection, args.query, embedder, args.k)
    finally:
        client.close()
    print(f"{args.collection}, {embedder.model_name}: {args.query}\n")
    for rank, hit in enumerate(hits, start=1):
        c = hit.chunk
        snippet = " ".join(c.text.split())
        snippet = snippet if len(snippet) <= SNIPPET else snippet[: SNIPPET - 3] + "..."
        print(f"{rank}. {hit.score:.3f}  {c.chunk_id}  [{c.language}, {c.doc_type}]  {c.section}")
        print(f"   {c.title}")
        print(f"   {snippet}\n")


if __name__ == "__main__":
    main()
