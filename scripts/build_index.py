"""Embed the chunk files and (re)build the Qdrant index (S2-06).

Reads the chunk files named in config.toml [index] (run `make ingest` first when chunking
changes), embeds them with the [embedding] model, and writes one collection per chunking
strategy to data/index/qdrant/ (gitignored). Writes data/index/index_build.json: model,
prefixes, the hash of each chunks file, point counts and truncation, so a build can be traced.
Usage: uv run python scripts/build_index.py
"""

import hashlib
import json
import time
from importlib.metadata import version

from qdrant_client import QdrantClient

from factory_docs.config import ROOT, load_config
from factory_docs.embedding import FastEmbedder
from factory_docs.index import build_collection, load_chunks

RECORD = ROOT / "data" / "index" / "index_build.json"


def file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = load_config()
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
        "store": config.index.path,
        "versions": {name: version(name) for name in ("fastembed", "onnxruntime", "qdrant-client")},
        "collections": {},
    }
    client = QdrantClient(path=str(store))
    try:
        for name, chunks_file in config.index.collections.items():
            path = ROOT / chunks_file
            chunks = load_chunks(path)
            lengths = embedder.token_lengths([chunk.text for chunk in chunks])
            limit = embedder.token_limit
            start = time.perf_counter()
            points = build_collection(client, name, chunks, embedder)
            seconds = time.perf_counter() - start
            truncated = sum(limit is not None and n > limit for n in lengths)
            record["collections"][name] = {
                "chunks_file": chunks_file,
                "chunks_sha256": file_sha256(path),
                "points": points,
                "truncated_chunks": truncated,
                "build_s": round(seconds, 1),
            }
            print(f"{name}: {points} points in {seconds:.0f} s, {truncated} chunks truncated")
    finally:
        client.close()
    RECORD.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Index: {config.index.path}; build record: {RECORD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
