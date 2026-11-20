"""The search index, S2-06 to S2-08: chunks in Qdrant (local mode, decision 13).

Qdrant in local mode runs inside the Python process and stores to a folder, like SQLite. One
collection per chunking strategy. Each point is one chunk: a stable ID derived from chunk_id,
two named vectors and the chunk's metadata and text as payload. "dense" is the embedding
(S2-06); "bm25" is a sparse vector of BM25 term weights, scored with Qdrant's IDF (S2-07,
`factory_docs.lexical`). A build drops the collection and recreates it, so it never duplicates.

Three ways to search: vector, BM25, and hybrid, which runs both and merges the two ranked
lists with reciprocal rank fusion (RRF). RRF uses only ranks, because cosine similarities and
BM25 scores are on different scales.

S2-08 adds two things. Filters (`SearchFilter`) on document type, language, date and latest
version: Qdrant applies them inside each retriever, before ranking, so a filtered search still
returns k hits. They use three payload fields derived at build time (`filter_fields`). And
reranking: a cross-encoder rescores the first-stage hits (`rerank`, `factory_docs.rerank`).
"""

import re
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from qdrant_client import QdrantClient, models

from factory_docs.chunking import Chunk
from factory_docs.config import SearchMode, SearchSettings
from factory_docs.embedding import Embedder
from factory_docs.lexical import Bm25Encoder, encode_query
from factory_docs.rerank import Reranker

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


MONTH_NAMES = (
    "january february march april may june july august september october november december"
)
MONTHS = {name: number for number, name in enumerate(MONTH_NAMES.split(), start=1)}
FULL_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
YEAR_MONTH = re.compile(r"\b(\d{4})-(\d{2})\b")
MONTH_YEAR = re.compile(r"\b(" + "|".join(MONTHS) + r") (\d{4})\b", re.IGNORECASE)
YEAR = re.compile(r"\b((?:19|20)\d{2})\b")


def parse_doc_date(version_or_date: str) -> str | None:
    """The document's date as YYYY-MM-DD, from the manifest's free-text version_or_date.

    The most precise form wins: a full date, else year-month (2015-11), else month name and
    year (February 2021), else a year (2014). A partial date counts as its first day. None
    when there is no date ("no date"), so a date filter leaves the document out.
    """
    if m := FULL_DATE.search(version_or_date):
        return date(int(m[1]), int(m[2]), int(m[3])).isoformat()
    if m := YEAR_MONTH.search(version_or_date):
        return date(int(m[1]), int(m[2]), 1).isoformat()
    if m := MONTH_YEAR.search(version_or_date):
        return date(int(m[2]), MONTHS[m[1].lower()], 1).isoformat()
    if m := YEAR.search(version_or_date):
        return date(int(m[1]), 1, 1).isoformat()
    return None


def filter_fields(chunk: Chunk, superseded: set[str]) -> dict:
    """Payload fields that only filters use; `Chunk` ignores them when a hit is read back.

    is_latest: no other document in the collection supersedes this one. languages: the
    document's languages, so a de filter also finds de/en documents (mul stays mul).
    """
    return {
        "is_latest": chunk.doc_id not in superseded,
        "doc_date": parse_doc_date(chunk.version_or_date),
        "languages": chunk.language.split("/"),
    }


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
    superseded = {chunk.supersedes for chunk in chunks if chunk.supersedes}
    points = [
        models.PointStruct(
            id=point_id(chunk.chunk_id),
            vector={DENSE: vector, BM25: bm25.encode_passage(chunk.text)},
            payload=chunk.model_dump() | filter_fields(chunk, superseded),
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    for start in range(0, len(points), UPSERT_BATCH):
        client.upsert(name, points=points[start : start + UPSERT_BATCH])
    return client.count(name, exact=True).count


@dataclass(frozen=True)
class SearchFilter:
    """Which chunks a search may return; conditions combine with AND, empty ones are ignored."""

    doc_types: tuple[str, ...] = ()  # any of these doc_type values
    languages: tuple[str, ...] = ()  # any of these languages (a de/en document is de and en)
    date_from: date | None = None  # document date on or after; undated documents drop out
    date_to: date | None = None  # document date on or before; undated documents drop out
    latest_only: bool = False  # leave out documents that a newer version supersedes

    def to_qdrant(self) -> models.Filter | None:
        must: list[models.Condition] = []
        if self.doc_types:
            must.append(
                models.FieldCondition(
                    key="doc_type", match=models.MatchAny(any=list(self.doc_types))
                )
            )
        if self.languages:
            must.append(
                models.FieldCondition(
                    key="languages", match=models.MatchAny(any=list(self.languages))
                )
            )
        if self.date_from or self.date_to:
            must.append(
                models.FieldCondition(
                    key="doc_date",
                    range=models.DatetimeRange(gte=self.date_from, lte=self.date_to),
                )
            )
        if self.latest_only:
            must.append(models.FieldCondition(key="is_latest", match=models.MatchValue(value=True)))
        return models.Filter(must=must) if must else None

    def describe(self) -> str:
        parts = []
        if self.doc_types:
            parts.append("type " + "|".join(self.doc_types))
        if self.languages:
            parts.append("language " + "|".join(self.languages))
        if self.date_from:
            parts.append(f"since {self.date_from}")
        if self.date_to:
            parts.append(f"until {self.date_to}")
        if self.latest_only:
            parts.append("latest versions only")
        return ", ".join(parts) or "none"


NO_FILTER = SearchFilter()


@dataclass(frozen=True)
class Hit:
    score: float  # cosine (vector), BM25 (0 up), RRF (hybrid) or reranker; higher is better
    chunk: Chunk
    vector_rank: int | None = None  # 1-based rank in the vector list; None: not in it
    bm25_rank: int | None = None  # 1-based rank in the BM25 list; None: not in it
    first_stage_rank: int | None = None  # rank before reranking; None: not reranked


def vector_search(
    client: QdrantClient,
    name: str,
    query: str,
    embedder: Embedder,
    k: int,
    search_filter: SearchFilter = NO_FILTER,
) -> list[Hit]:
    """The k chunks whose vectors are closest to the query's."""
    response = client.query_points(
        name,
        query=embedder.embed_query(query),
        using=DENSE,
        limit=k,
        query_filter=search_filter.to_qdrant(),
        with_payload=True,
    )
    return [
        Hit(point.score, Chunk.model_validate(point.payload), vector_rank=rank)
        for rank, point in enumerate(response.points, start=1)
    ]


def lexical_search(
    client: QdrantClient,
    name: str,
    query: str,
    k: int,
    search_filter: SearchFilter = NO_FILTER,
) -> list[Hit]:
    """The k chunks with the highest BM25 score; only chunks sharing a query term score."""
    vector = encode_query(query)
    if not vector.indices:  # no words in the query, e.g. "?"
        return []
    response = client.query_points(
        name,
        query=vector,
        using=BM25,
        limit=k,
        query_filter=search_filter.to_qdrant(),
        with_payload=True,
    )
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
    search_filter: SearchFilter = NO_FILTER,
) -> list[Hit]:
    """Vector and BM25 search, prefetch_k candidates each, merged by RRF into the top k."""
    dense = vector_search(client, name, query, embedder, prefetch_k, search_filter)
    lexical = lexical_search(client, name, query, prefetch_k, search_filter)
    chunks = {hit.chunk.chunk_id: hit.chunk for hit in [*dense, *lexical]}
    vector_rank = {hit.chunk.chunk_id: hit.vector_rank for hit in dense}
    bm25_rank = {hit.chunk.chunk_id: hit.bm25_rank for hit in lexical}
    fused = rrf([[hit.chunk.chunk_id for hit in hits] for hits in (dense, lexical)], rrf_k)
    return [
        Hit(score, chunks[chunk_id], vector_rank.get(chunk_id), bm25_rank.get(chunk_id))
        for chunk_id, score in fused[:k]
    ]


def rerank(query: str, hits: Sequence[Hit], reranker: Reranker, k: int) -> list[Hit]:
    """Rescore `hits` with the cross-encoder and keep the best k.

    The reranker score replaces the first-stage score; first_stage_rank keeps where the hit
    was before. Equal scores keep the first-stage order. Only the given hits can come back.
    """
    scores = reranker.score(query, [hit.chunk.text for hit in hits])
    if len(scores) != len(hits):
        raise ValueError(f"reranker returned {len(scores)} scores for {len(hits)} hits")
    order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
    return [replace(hits[i], score=scores[i], first_stage_rank=i + 1) for i in order[:k]]


def search(
    client: QdrantClient,
    name: str,
    query: str,
    embedder: Embedder | None,
    k: int,
    settings: SearchSettings,
    mode: SearchMode | None = None,
    search_filter: SearchFilter = NO_FILTER,
    reranker: Reranker | None = None,
    candidates: int | None = None,
) -> list[Hit]:
    """Search in `mode` (default: settings.mode); BM25 alone needs no embedder.

    With a reranker, the first stage returns max(k, candidates) hits, and the reranker keeps
    the best k of them.
    """
    mode = mode or settings.mode
    first_k = max(k, candidates or k) if reranker else k
    if mode == "bm25":
        hits = lexical_search(client, name, query, first_k, search_filter)
    elif embedder is None:
        raise ValueError(f"{mode} search needs an embedder")
    elif mode == "vector":
        hits = vector_search(client, name, query, embedder, first_k, search_filter)
    else:
        hits = hybrid_search(
            client,
            name,
            query,
            embedder,
            first_k,
            settings.prefetch_k,
            settings.rrf_k,
            search_filter,
        )
    return rerank(query, hits, reranker, k) if reranker else hits
