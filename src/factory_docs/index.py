"""The search index, S2-06 and S2-07: chunks in Qdrant (local mode, decision 13).

Qdrant in local mode runs inside the Python process and stores to a folder, like SQLite. One
collection per chunking strategy. Each point is one chunk: a stable ID derived from chunk_id,
two named vectors and the chunk's metadata and text as payload. "dense" is the embedding
(S2-06); "bm25" is a sparse vector of BM25 term weights, scored with Qdrant's IDF (S2-07,
`factory_docs.lexical`). A build drops the collection and recreates it, so it never duplicates.

Three ways to search: vector, BM25, and hybrid, which runs both and merges the two ranked
lists with reciprocal rank fusion (RRF). RRF uses only ranks, because cosine similarities and
BM25 scores are on different scales.
"""

import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient, models

from factory_docs.chunking import Chunk
from factory_docs.config import SearchMode, SearchSettings
from factory_docs.embedding import Embedder
from factory_docs.lexical import Bm25Encoder, encode_query

DENSE = "dense"
BM25 = "bm25"
UPSERT_BATCH = 256
# Fixed namespace, so a chunk_id maps to the same point ID on every machine and every build
POINT_NAMESPACE = uuid.UUID("5d0f6a8e-3c1b-4f5e-9a57-6b2a1d4c8e90")


def load_chunks(path: Path) -> list[Chunk]:
    """Every line of a chunks file, validated; a bad line stops the build."""
    chunks = []
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            try:
                chunks.append(Chunk.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"{path.name} line {number}: {error}") from error
    ids = [chunk.chunk_id for chunk in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path.name}: duplicate chunk_id")
    return chunks


def point_id(chunk_id: str) -> str:
    """Qdrant takes integers or UUIDs as point IDs, not strings like D001-s-0007."""
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def build_collection(
    client: QdrantClient,
    name: str,
    chunks: Sequence[Chunk],
    embedder: Embedder,
    bm25: Bm25Encoder,
) -> int:
    """(Re)create collection `name` from `chunks`; returns the number of points.

    `bm25` must be fitted on the same chunks, so its average length is this collection's.
    """
    vectors = embedder.embed_passages([chunk.text for chunk in chunks])
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config={
            DENSE: models.VectorParams(size=embedder.dimension, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={BM25: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    points = [
        models.PointStruct(
            id=point_id(chunk.chunk_id),
            vector={DENSE: vector, BM25: bm25.encode_passage(chunk.text)},
            payload=chunk.model_dump(),
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    for start in range(0, len(points), UPSERT_BATCH):
        client.upsert(name, points=points[start : start + UPSERT_BATCH])
    return client.count(name, exact=True).count


@dataclass(frozen=True)
class Hit:
    score: float  # cosine (vector, -1 to 1), BM25 (0 up) or RRF (hybrid); higher is better
    chunk: Chunk
    vector_rank: int | None = None  # 1-based rank in the vector list; None: not in it
    bm25_rank: int | None = None  # 1-based rank in the BM25 list; None: not in it


def vector_search(
    client: QdrantClient, name: str, query: str, embedder: Embedder, k: int
) -> list[Hit]:
    """The k chunks whose vectors are closest to the query's."""
    response = client.query_points(
        name, query=embedder.embed_query(query), using=DENSE, limit=k, with_payload=True
    )
    return [
        Hit(point.score, Chunk.model_validate(point.payload), vector_rank=rank)
        for rank, point in enumerate(response.points, start=1)
    ]


def lexical_search(client: QdrantClient, name: str, query: str, k: int) -> list[Hit]:
    """The k chunks with the highest BM25 score; only chunks sharing a query term score."""
    vector = encode_query(query)
    if not vector.indices:  # no words in the query, e.g. "?"
        return []
    response = client.query_points(name, query=vector, using=BM25, limit=k, with_payload=True)
    return [
        Hit(point.score, Chunk.model_validate(point.payload), bm25_rank=rank)
        for rank, point in enumerate(response.points, start=1)
    ]


def rrf(rankings: Sequence[Sequence[str]], k: int) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: each list adds 1 / (k + rank) for every item in it.

    An item high in either list ends up high; one in both lists gets both shares. A larger k
    flattens the difference between rank 1 and rank 10. Ties go by key, so the order is stable.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def hybrid_search(
    client: QdrantClient,
    name: str,
    query: str,
    embedder: Embedder,
    k: int,
    prefetch_k: int,
    rrf_k: int,
) -> list[Hit]:
    """Vector and BM25 search, prefetch_k candidates each, merged by RRF into the top k."""
    dense = vector_search(client, name, query, embedder, prefetch_k)
    lexical = lexical_search(client, name, query, prefetch_k)
    chunks = {hit.chunk.chunk_id: hit.chunk for hit in [*dense, *lexical]}
    vector_rank = {hit.chunk.chunk_id: hit.vector_rank for hit in dense}
    bm25_rank = {hit.chunk.chunk_id: hit.bm25_rank for hit in lexical}
    fused = rrf([[hit.chunk.chunk_id for hit in hits] for hits in (dense, lexical)], rrf_k)
    return [
        Hit(score, chunks[chunk_id], vector_rank.get(chunk_id), bm25_rank.get(chunk_id))
        for chunk_id, score in fused[:k]
    ]


def search(
    client: QdrantClient,
    name: str,
    query: str,
    embedder: Embedder | None,
    k: int,
    settings: SearchSettings,
    mode: SearchMode | None = None,
) -> list[Hit]:
    """Search in `mode` (default: settings.mode); BM25 alone needs no embedder."""
    mode = mode or settings.mode
    if mode == "bm25":
        return lexical_search(client, name, query, k)
    if embedder is None:
        raise ValueError(f"{mode} search needs an embedder")
    if mode == "vector":
        return vector_search(client, name, query, embedder, k)
    return hybrid_search(client, name, query, embedder, k, settings.prefetch_k, settings.rrf_k)
