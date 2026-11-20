"""Search the index from the command line: vector, BM25 or hybrid, optionally reranked and
filtered (S2-06 to S2-08).

Each hit shows its rank in the vector list and in the BM25 list ("-" when it is not in the
first prefetch_k of that list), so you can see which retriever found it; after reranking, also
its rank before reranking.
Usage: uv run python scripts/search.py "question" [--collection structure|fixed] [--k 5]
       [--mode vector|bm25|hybrid] [--rerank | --no-rerank] [--doc-type sds,manual]
       [--language uk,de] [--since 2024-01-01] [--until 2025-12-31] [--latest-only]
"""

import argparse
from datetime import date
from typing import get_args

from qdrant_client import QdrantClient

from factory_docs.config import ROOT, SearchMode, load_config
from factory_docs.embedding import FastEmbedder
from factory_docs.index import BM25, Hit, SearchFilter, search
from factory_docs.rerank import FastReranker

SNIPPET = 150
FILTER_FIELDS = "is_latest"  # a payload field every index built since S2-08 has


def ranks(hit: Hit) -> str:
    def show(rank: int | None) -> str:
        return "-" if rank is None else str(rank)

    text = f"vector {show(hit.vector_rank)}, bm25 {show(hit.bm25_rank)}"
    if hit.first_stage_rank is not None:
        text += f", before reranking {hit.first_stage_rank}"
    return text


def values(text: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in text.split(",") if v.strip())


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("--collection", default="structure", choices=list(config.index.collections))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=get_args(SearchMode), default=config.search.mode)
    parser.add_argument(
        "--rerank", action=argparse.BooleanOptionalAction, default=config.search.rerank
    )
    parser.add_argument("--doc-type", type=values, default=(), help="comma-separated doc_type")
    parser.add_argument("--language", type=values, default=(), help="comma-separated languages")
    parser.add_argument("--since", type=date.fromisoformat, help="document date on or after")
    parser.add_argument("--until", type=date.fromisoformat, help="document date on or before")
    parser.add_argument("--latest-only", action="store_true", help="skip superseded versions")
    args = parser.parse_args()
    search_filter = SearchFilter(
        doc_types=args.doc_type,
        languages=args.language,
        date_from=args.since,
        date_to=args.until,
        latest_only=args.latest_only,
    )

    store = ROOT / config.index.path
    if not store.exists():
        raise SystemExit(f"No index at {config.index.path}; run `make index` first")
    embedder = None if args.mode == "bm25" else FastEmbedder.from_config(config)
    reranker = FastReranker.from_config(config) if args.rerank else None
    client = QdrantClient(path=str(store))
    try:
        if BM25 not in (client.get_collection(args.collection).config.params.sparse_vectors or {}):
            raise SystemExit("The index has no BM25 vectors (built before S2-07); run `make index`")
        sample = client.scroll(args.collection, limit=1, with_payload=True)[0]
        if sample and FILTER_FIELDS not in (sample[0].payload or {}):
            raise SystemExit(
                "The index has no filter fields (built before S2-08); run `make index`"
            )
        hits = search(
            client,
            args.collection,
            args.query,
            embedder,
            args.k,
            config.search,
            args.mode,
            search_filter,
            reranker,
            config.reranking.candidates,
        )
    finally:
        client.close()
    model = f", {embedder.model_name}" if embedder else ""
    if reranker:
        model += f", reranked by {reranker.model_name} ({config.reranking.candidates} candidates)"
    print(f"{args.collection}, {args.mode}{model}: {args.query}")
    print(f"Filter: {search_filter.describe()}\n")
    if not hits:
        print("No hits.")
    for rank, hit in enumerate(hits, start=1):
        c = hit.chunk
        snippet = " ".join(c.text.split())
        snippet = snippet if len(snippet) <= SNIPPET else snippet[: SNIPPET - 3] + "..."
        print(f"{rank}. {hit.score:.4f}  {c.chunk_id}  [{c.language}, {c.doc_type}]  {c.section}")
        print(f"   {c.title}  ({c.version_or_date}; {ranks(hit)})")
        print(f"   {snippet}\n")


if __name__ == "__main__":
    main()
