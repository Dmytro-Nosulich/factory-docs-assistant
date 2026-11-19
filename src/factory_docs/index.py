"""The vector index, S2-06: chunks and their embeddings in Qdrant (local mode, decision 13).

Qdrant in local mode runs inside the Python process and stores to a folder, like SQLite. One
collection per chunking strategy. Each point is one chunk: a stable ID derived from chunk_id,
a named vector "dense" (S2-07 adds a sparse BM25 vector next to it) and the chunk's metadata
and text as payload. A build drops the collection and recreates it, so it never duplicates.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient, models

from factory_docs.chunking import Chunk
from factory_docs.embedding import Embedder

DENSE = "dense"
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
    client: QdrantClient, name: str, chunks: Sequence[Chunk], embedder: Embedder
) -> int:
    """(Re)create collection `name` from `chunks`; returns the number of points."""
    vectors = embedder.embed_passages([chunk.text for chunk in chunks])
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config={
            DENSE: models.VectorParams(size=embedder.dimension, distance=models.Distance.COSINE)
        },
    )
    points = [
        models.PointStruct(
            id=point_id(chunk.chunk_id), vector={DENSE: vector}, payload=chunk.model_dump()
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    for start in range(0, len(points), UPSERT_BATCH):
        client.upsert(name, points=points[start : start + UPSERT_BATCH])
    return client.count(name, exact=True).count


@dataclass(frozen=True)
class Hit:
    score: float  # cosine similarity, -1 to 1; higher is closer
    chunk: Chunk


def search(client: QdrantClient, name: str, query: str, embedder: Embedder, k: int) -> list[Hit]:
    """The k chunks whose vectors are closest to the query's."""
    response = client.query_points(
        name, query=embedder.embed_query(query), using=DENSE, limit=k, with_payload=True
    )
    return [Hit(point.score, Chunk.model_validate(point.payload)) for point in response.points]
