"""Embed the chunk files and (re)build the Qdrant index (S2-06; BM25 S2-07; filter fields S2-08).

Reads the chunk files named in config.toml [index] (run `make ingest` first when chunking
changes), embeds them with the [embedding] model, adds BM25 sparse vectors with the [search]
k1 and b, and writes one collection per chunking strategy to data/index/qdrant/ (gitignored).
Writes data/index/index_build.json: model, prefixes, BM25 parameters, the hash of each chunks
file, point counts, truncation, average chunk length in BM25 tokens and vocabulary size, and
which documents the filters treat as superseded or undated, so a build can be traced.
Usage: uv run python scripts/build_index.py [--if-stale]
  --if-stale  rebuild only when `factory_docs.index.stale_reasons` finds a reason (`make run-rag`,
              S2-15); without it, always rebuild (`make index`)
"""

import argparse
import json
import time

from qdrant_client import QdrantClient

from factory_docs.config import ROOT, Config, load_config
from factory_docs.embedding import FastEmbedder
from factory_docs.index import (
    build_collection,
    build_inputs,
    file_sha256,
    load_chunks,
    parse_doc_date,
    stale_reasons,
)
from factory_docs.lexical import Bm25Encoder, vocabulary

RECORD = ROOT / "data" / "index" / "index_build.json"


def store_points(config: Config) -> dict[str, int] | None:
    """Point count per collection in the store, or None when there is no store folder."""
    store = ROOT / config.index.path
    if not store.exists():
        return None
    client = QdrantClient(path=str(store))
    try:
        names = [c.name for c in client.get_collections().collections]
        return {name: client.count(name, exact=True).count for name in names}
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--if-stale", action="store_true", help="skip when the index is current")
    args = parser.parse_args()
    config = load_config()
    if args.if_stale:
        saved = json.loads(RECORD.read_text(encoding="utf-8")) if RECORD.exists() else None
        reasons = stale_reasons(saved, build_inputs(config), store_points(config))
        if not reasons:
            print(f"Index {config.index.path} is current ({RECORD.relative_to(ROOT)}); skipped")
            return
        print("Rebuilding the index: " + "; ".join(reasons))
    key = config.embedding.model
    embedder = FastEmbedder.from_config(config, key)
    model = config.embedding_models[key]
    store = ROOT / config.index.path
    store.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "embedding": {
            "key": key,
            "model": model.name,
            "dimension": embedder.dimension,
            "query_prefix": model.query_prefix,
            "passage_prefix": model.passage_prefix,
            "token_limit": embedder.token_limit,
            "batch_size": config.embedding.batch_size,
        },
        "bm25": {"k1": config.search.bm25_k1, "b": config.search.bm25_b},
        "store": config.index.path,
        "versions": build_inputs(config)["versions"],
        "collections": {},
    }
    client = QdrantClient(path=str(store))
    try:
        for name, chunks_file in config.index.collections.items():
            path = ROOT / chunks_file
            chunks = load_chunks(path)
            texts = [chunk.text for chunk in chunks]
            lengths = embedder.token_lengths(texts)
            limit = embedder.token_limit
            bm25 = Bm25Encoder.fit(texts, config.search.bm25_k1, config.search.bm25_b)
            terms, colliding = vocabulary(texts)
            start = time.perf_counter()
            points = build_collection(client, name, chunks, embedder, bm25)
            seconds = time.perf_counter() - start
            truncated = sum(limit is not None and n > limit for n in lengths)
            superseded = sorted({c.supersedes for c in chunks if c.supersedes})
            undated = sorted(
                {c.doc_id for c in chunks if parse_doc_date(c.version_or_date) is None}
            )
            record["collections"][name] = {
                "chunks_file": chunks_file,
                "chunks_sha256": file_sha256(path),
                "points": points,
                "truncated_chunks": truncated,
                "bm25_avg_tokens": round(bm25.avg_len, 2),
                "bm25_vocabulary": terms,
                "bm25_colliding_terms": colliding,
                "documents": len({c.doc_id for c in chunks}),
                "superseded_documents": superseded,  # is_latest false (S2-08)
                "undated_documents": undated,  # doc_date null: a date filter leaves them out
                "build_s": round(seconds, 1),
            }
            print(
                f"{name}: {points} points in {seconds:.0f} s, {truncated} chunks truncated, "
                f"{terms} BM25 terms ({colliding} share a term ID), superseded {superseded}, "
                f"undated {undated}"
            )
    finally:
        client.close()
    RECORD.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Index: {config.index.path}; build record: {RECORD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
